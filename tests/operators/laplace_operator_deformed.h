// Portable::LaplaceOperator on a curved, sheared mesh: every vmult variant
// that runs on host and device builds, Tvmult() and compute_diagonal()
// against the CPU FEValues reference of deformed_mesh_util.h; with
// batched_copy only vmult_batched() and vmult_fused() (the batched
// FEEvaluation copied from deal.II, which needs PrecomputedData::first_cell).
//
// src is zero on the Dirichlet DoFs: the deal.II cell_loop paths read src
// there, this project's own kernels do not.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "deformed_mesh_util.h"
#include "operators/portable_laplace_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, bool batched_copy = false>
void
test(const bool overlap)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Operator   = Portable::LaplaceOperator<dim, fe_degree, Number>;

  constexpr double tolerance = 1e-12;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FE_Q<dim>     fe(fe_degree);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(), constraints);
  constraints.close();

  Operator op(dof_handler, constraints, overlap);

  VectorType src, dst, dst_ref, diag_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_ref);

  fill_source(src, locally_owned_dofs, &constraints);

  reference_laplace<dim, Number>(mapping,
                                 dof_handler,
                                 constraints,
                                 fe_degree + 1,
                                 locally_owned_dofs,
                                 locally_relevant_dofs,
                                 MPI_COMM_WORLD,
                                 src,
                                 dst_ref,
                                 diag_ref);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  const auto check = [&](const std::string &name, auto &&apply) {
    dst = 0.;
    apply(dst, src);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);
  };

  if constexpr (batched_copy)
    {
      check("vmult_batched", [&](VectorType &d, const VectorType &s) { op.vmult_batched(d, s); });
      check("vmult_fused", [&](VectorType &d, const VectorType &s) { op.vmult_fused(d, s); });
      return;
    }

  check("vmult", [&](VectorType &d, const VectorType &s) { op.vmult(d, s); });
  check("Tvmult", [&](VectorType &d, const VectorType &s) { op.Tvmult(d, s); });
  check("vmult_dealii", [&](VectorType &d, const VectorType &s) { op.vmult_dealii(d, s); });
  check("vmult_bk3", [&](VectorType &d, const VectorType &s) { op.vmult_bk3(d, s); });
  check("vmult_bk3_abstracted",
        [&](VectorType &d, const VectorType &s) { op.vmult_bk3_abstracted(d, s); });
  check("vmult_dealii_batched",
        [&](VectorType &d, const VectorType &s) { op.vmult_dealii_batched(d, s); });
  check("vmult_dealii_batched_fused",
        [&](VectorType &d, const VectorType &s) { op.vmult_dealii_batched_fused(d, s); });
  check("vmult_dealii_batched_view",
        [&](VectorType &d, const VectorType &s) { op.vmult_dealii_batched_view(d, s); });

  op.compute_diagonal();
  Kokkos::fence();
  report("compute_diagonal",
         max_relative_diagonal_error(op.get_matrix_diagonal_inverse()->get_vector(),
                                     diag_ref,
                                     locally_owned_dofs,
                                     MPI_COMM_WORLD),
         tolerance);
}



template <bool batched_copy = false>
void
test_all(const bool overlap)
{
  test<2, 1, batched_copy>(overlap);
  test<2, 2, batched_copy>(overlap);
  test<2, 3, batched_copy>(overlap);
  test<2, 4, batched_copy>(overlap);
  test<3, 1, batched_copy>(overlap);
  test<3, 2, batched_copy>(overlap);
  test<3, 3, batched_copy>(overlap);
  test<3, 4, batched_copy>(overlap);
}
