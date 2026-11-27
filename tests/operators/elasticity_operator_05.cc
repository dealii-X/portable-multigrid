// Portable::ElasticityOperator with the Saint Venant-Kirchhoff material: the
// nonlinear residual R(u) by vmult, vmult_dealii, vmult_batched and
// vmult_fused on the curved, sheared mesh of deformed_mesh_util.h against
// the CPU FEValues reference. u is scaled so that the strains are large
// enough for the nonlinear part to matter, which the comparison with the
// linear reference confirms.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_elasticity_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  constexpr double tolerance = 1e-12;

  const Material material{1.7, 0.6};

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FESystem<dim> fe(FE_Q<dim>{fe_degree}, dim);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints);
  constraints.close();

  const Operator op(dof_handler, constraints, material, false);

  VectorType src, dst, dst_ref, dst_linear, diag_linear;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(dst_linear);
  op.initialize_dof_vector(diag_linear);

  fill_source(src, locally_owned_dofs, &constraints);
  src *= 0.05;

  reference_svk_residual<dim, Number>(mapping,
                                      dof_handler,
                                      constraints,
                                      n_q_points_1d,
                                      material.lambda,
                                      material.mu,
                                      locally_owned_dofs,
                                      locally_relevant_dofs,
                                      MPI_COMM_WORLD,
                                      src,
                                      dst_ref);
  reference_elasticity<dim, Number>(mapping,
                                    dof_handler,
                                    constraints,
                                    n_q_points_1d,
                                    material.lambda,
                                    material.mu,
                                    locally_owned_dofs,
                                    locally_relevant_dofs,
                                    MPI_COMM_WORLD,
                                    src,
                                    dst_linear,
                                    diag_linear);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
  deallog << "nonlinear part above 1e-2: " << std::boolalpha
          << (relative_l2_difference(dst_linear, dst_ref) > 1e-2) << std::endl;

  const auto check = [&](const std::string &name, auto &&apply)
  {
    dst = 0.;
    apply(dst, src);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);
  };

  check("vmult", [&](VectorType &d, const VectorType &s) { op.vmult(d, s); });
  check("vmult_dealii", [&](VectorType &d, const VectorType &s) { op.vmult_dealii(d, s); });
  check("vmult_batched", [&](VectorType &d, const VectorType &s) { op.vmult_batched(d, s); });
  check("vmult_fused", [&](VectorType &d, const VectorType &s) { op.vmult_fused(d, s); });
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>();
  test<2, 2>();
  test<2, 3>();
  test<3, 1>();
  test<3, 2>();
  test<3, 3>();
  // fe_degree + 2 points
  test<3, 2, 4>();
  // general kernel: too many points for the transformation to collocation
  test<3, 1, 3>();
}
