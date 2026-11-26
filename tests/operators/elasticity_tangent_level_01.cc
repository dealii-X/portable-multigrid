// Portable::ElasticityTangentLevelOperator with the Saint Venant-Kirchhoff
// material on the curved, sheared mesh of deformed_mesh_util.h, with zero
// Dirichlet values on component 1 (constraints) and prescribed DoFs on
// component 0 at the boundary:
// - vmult() = D A D + S against the CPU FEValues reference of A at u
// - the same on the level of degree 1, linearized at the interpolation of u
// - the inverse diagonal against D diag(A) + S

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_elasticity_tangent_level_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

using Number     = double;
using VectorType = DeviceVectorType<Number>;

template <int dim>
struct Level
{
  Level(const Triangulation<dim> &triangulation, const unsigned int degree)
    : fe(FE_Q<dim>{degree}, dim)
    , mapping(degree)
    , dof_handler(triangulation)
  {
    dof_handler.distribute_dofs(fe);
    owned    = dof_handler.locally_owned_dofs();
    relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.reinit(owned, relevant);
    ComponentMask mask_1(dim, false);
    mask_1.set(1, true);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints, mask_1);
    constraints.close();

    ComponentMask mask_0(dim, false);
    mask_0.set(0, true);
    prescribed = DoFTools::extract_boundary_dofs(dof_handler, mask_0);
    prescribed = prescribed & owned;
  }

  // D src (prescribed = false) or S src (prescribed = true)
  void
  mask(VectorType &vec, const bool keep_prescribed) const
  {
    LinearAlgebra::ReadWriteVector<Number> rw(owned);
    rw.import_elements(vec, VectorOperation::insert);
    for (const auto i : owned)
      if (prescribed.is_element(i) != keep_prescribed)
        rw(i) = 0.;
    vec.import_elements(rw, VectorOperation::insert);
  }

  const FESystem<dim>       fe;
  const MappingQ<dim>       mapping;
  DoFHandler<dim>           dof_handler;
  IndexSet                  owned, relevant, prescribed;
  AffineConstraints<Number> constraints;
};

template <int dim, int degree, int degree_coarse>
void
test(const unsigned int n_refinements)
{
  using Material      = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using LevelOperator = Portable::ElasticityTangentLevelOperator<dim, degree, Number, Material>;
  using CoarseOperator =
    Portable::ElasticityTangentLevelOperator<dim, degree_coarse, Number, Material>;

  const Material material{1.7, 0.6};

  deallog << "dim = " << dim << ", degree = " << degree << ", degree_coarse = " << degree_coarse
          << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, n_refinements);

  const Level<dim> fine(triangulation, degree), coarse(triangulation, degree_coarse);

  // A D src + S src with the reference A at state, constrained rows src
  const auto reference =
    [&](const Level<dim> &level, const VectorType &state, const VectorType &src, VectorType &dst)
  {
    VectorType free_src(src), prescribed_src(src);
    level.mask(free_src, false);
    level.mask(prescribed_src, true);
    reference_svk_linearized<dim, Number>(level.mapping,
                                          level.dof_handler,
                                          level.constraints,
                                          level.fe.degree + 1,
                                          material.lambda,
                                          material.mu,
                                          level.owned,
                                          level.relevant,
                                          MPI_COMM_WORLD,
                                          state,
                                          free_src,
                                          dst);
    level.mask(dst, false);
    dst += prescribed_src;
  };

  LevelOperator  op(fine.dof_handler, fine.constraints, material, fine.prescribed);
  CoarseOperator op_coarse(coarse.dof_handler, coarse.constraints, material, coarse.prescribed);

  VectorType u, src, dst, dst_ref;
  for (VectorType *vec : {&u, &src, &dst, &dst_ref})
    op.initialize_dof_vector(*vec);

  fill_source(u, fine.owned, &fine.constraints);
  u *= 0.05;
  {
    LinearAlgebra::ReadWriteVector<Number> rw(fine.owned);
    for (const auto i : fine.owned)
      rw(i) = fine.constraints.is_constrained(i) ? 0. : std::cos(0.29 * static_cast<double>(i + 2));
    src.import_elements(rw, VectorOperation::insert);
  }

  deallog << "n_dofs = " << fine.dof_handler.n_dofs()
          << ", prescribed = " << Utilities::MPI::sum(fine.prescribed.n_elements(), MPI_COMM_WORLD)
          << std::endl;

  op.linearize(u);
  op.vmult(dst, src);
  reference(fine, u, src, dst_ref);
  report("vmult", relative_l2_difference(dst, dst_ref), 1e-12);

  // the inverse diagonal times D diag(A) + S
  {
    op.compute_diagonal();
    VectorType diagonal, ones;
    op.get_operator().compute_linearized_diagonal(diagonal);
    fine.mask(diagonal, false);
    op.initialize_dof_vector(ones);
    ones = 1.;
    VectorType prescribed_ones(ones);
    fine.mask(prescribed_ones, true);
    diagonal += prescribed_ones;
    diagonal.scale(op.get_matrix_diagonal_inverse()->get_vector());
    report("inverse diagonal", relative_l2_difference(diagonal, ones), 1e-14);
  }

  // coarse level at the interpolation of u
  {
    VectorType u_coarse, src_coarse, dst_coarse, dst_coarse_ref;
    for (VectorType *vec : {&u_coarse, &src_coarse, &dst_coarse, &dst_coarse_ref})
      op_coarse.initialize_dof_vector(*vec);
    Portable::interpolate_state(
      fine.dof_handler, u, coarse.dof_handler, coarse.constraints, u_coarse);
    fill_source(src_coarse, coarse.owned, &coarse.constraints);

    op_coarse.linearize(u_coarse);
    op_coarse.vmult(dst_coarse, src_coarse);
    reference(coarse, u_coarse, src_coarse, dst_coarse_ref);
    report("coarse vmult", relative_l2_difference(dst_coarse, dst_coarse_ref), 1e-12);
  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 2, 1>(3);
  test<3, 2, 1>(2);
  test<3, 4, 2>(1);
}
