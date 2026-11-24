// Portable::ElasticityOperator (FESystem of dim FE_Q) on a curved, sheared
// mesh: vmult, Tvmult, vmult_dealii, vmult_batched and compute_diagonal()
// against the CPU FEValues reference of deformed_mesh_util.h.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "deformed_mesh_util.h"
#include "operators/portable_elasticity_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test(const bool overlap)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::LinearElasticMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  constexpr double tolerance = 1e-12;

  // different, so that neither term can hide the other
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

  Operator op(dof_handler, constraints, material, overlap);

  VectorType src, dst, dst_ref, diag_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_ref);

  fill_source(src, locally_owned_dofs, &constraints);

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
                                    dst_ref,
                                    diag_ref);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  const auto check = [&](const std::string &name, auto &&apply)
  {
    dst = 0.;
    apply(dst, src);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);
  };

  check("vmult", [&](VectorType &d, const VectorType &s) { op.vmult(d, s); });
  check("Tvmult", [&](VectorType &d, const VectorType &s) { op.Tvmult(d, s); });
  check("vmult_dealii", [&](VectorType &d, const VectorType &s) { op.vmult_dealii(d, s); });
  check("vmult_batched", [&](VectorType &d, const VectorType &s) { op.vmult_batched(d, s); });

  op.compute_diagonal();
  Kokkos::fence();
  report("compute_diagonal",
         max_relative_diagonal_error(op.get_matrix_diagonal_inverse()->get_vector(),
                                     diag_ref,
                                     locally_owned_dofs,
                                     MPI_COMM_WORLD),
         tolerance);
}



inline void
test_all(const bool overlap)
{
  test<2, 1>(overlap);
  test<2, 2>(overlap);
  test<2, 3>(overlap);
  test<3, 1>(overlap);
  test<3, 2>(overlap);
  test<3, 3>(overlap);
  // fe_degree + 2 points, the quadrature of the benchmark kernel
  test<2, 2, 4>(overlap);
  test<3, 2, 4>(overlap);
}
