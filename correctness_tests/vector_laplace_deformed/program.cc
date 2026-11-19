// Deformed-mesh correctness check for Portable::VectorLaplaceOperator
// (portable_vector_laplace_operator.h): vmult() (= vmult_dealii()),
// vmult_bk4(), Tvmult() and compute_diagonal() are compared against the
// component-wise CPU FEValues reference of
// correctness_tests/common/deformed_mesh_test_utils.h on a curved, sheared
// mesh with MappingQ(fe_degree). The FE is FESystem(FE_Q(p), n_components).
//
// As in correctness_tests/vector_laplace_bk4, src is zero on the Dirichlet
// DoFs: the deal.II cell_loop path reads them, the BK4 kernel masks them.
//
// Prints one line per variant and PASSED/FAILED with exit code 0/1.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>

#include "deformed_mesh_test_utils.h"
#include "operators/portable_vector_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;
  using namespace deformed_mesh_test;

  template <int dim, int fe_degree, int n_components>
  double
  run_test(const MPI_Comm      mpi_communicator,
           ConditionalOStream &pcout,
           const bool          overlap,
           unsigned int       &n_skipped)
  {
    using Number     = double;
    using VectorType = DeviceVectorType<Number>;
    using Operator   = Portable::VectorLaplaceOperator<dim, fe_degree, n_components, Number>;

    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_components = " << n_components << ", overlap = " << (overlap ? "true" : "false")
          << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    const CurvedShearedManifold<dim>          manifold;
    make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

    const FE_Q<dim>     scalar_fe(fe_degree);
    const FESystem<dim> fe(scalar_fe, n_components);
    const MappingQ<dim> mapping(fe_degree);
    DoFHandler<dim>     dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<Number> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(n_components), constraints);
    constraints.close();

    const auto op = make_operator_or_skip<Operator>(pcout, overlap, dof_handler, constraints, overlap);
    if (!op)
      {
        ++n_skipped;
        return -1.;
      }

    VectorType src, dst, dst_ref, diag_ref;
    op->initialize_dof_vector(src);
    op->initialize_dof_vector(dst);
    op->initialize_dof_vector(dst_ref);
    op->initialize_dof_vector(diag_ref);

    fill_source(src, locally_owned_dofs, &constraints);

    reference_laplace<dim, Number>(mapping,
                                   dof_handler,
                                   constraints,
                                   fe_degree + 1,
                                   locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator,
                                   src,
                                   dst_ref,
                                   diag_ref);

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_ref| = " << dst_ref.l2_norm()
          << std::endl;

    Results results;

    const auto check = [&](const std::string &name, auto &&apply) {
      dst = 0.;
      apply(dst, src);
      Kokkos::fence();
      results.record(pcout, name, relative_l2_difference(dst, dst_ref));
    };

    check("vmult", [&](VectorType &d, const VectorType &s) { op->vmult(d, s); });
    check("Tvmult", [&](VectorType &d, const VectorType &s) { op->Tvmult(d, s); });
    check("vmult_dealii", [&](VectorType &d, const VectorType &s) { op->vmult_dealii(d, s); });
    check("vmult_bk4", [&](VectorType &d, const VectorType &s) { op->vmult_bk4(d, s); });

    op->compute_diagonal();
    Kokkos::fence();
    results.record(pcout,
                   "compute_diagonal",
                   max_relative_diagonal_error(op->get_matrix_diagonal_inverse()->get_vector(),
                                               diag_ref,
                                               locally_owned_dofs,
                                               mpi_communicator));

    pcout << std::endl;
    return results.max_err;
  }

  template <int dim, int fe_degree, int n_components>
  void
  run_both_overlap_modes(const MPI_Comm      mpi_communicator,
                         ConditionalOStream &pcout,
                         double             &max_err,
                         unsigned int       &n_skipped)
  {
    for (const bool overlap : {false, true})
      {
        const double err =
          run_test<dim, fe_degree, n_components>(mpi_communicator, pcout, overlap, n_skipped);
        if (err >= 0.)
          max_err = std::max(max_err, err);
      }
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double       max_err   = 0.;
  unsigned int n_skipped = 0;

  try
    {
      run_both_overlap_modes<3, 1, 3>(MPI_COMM_WORLD, pcout, max_err, n_skipped);
      run_both_overlap_modes<3, 2, 3>(MPI_COMM_WORLD, pcout, max_err, n_skipped);
      run_both_overlap_modes<3, 3, 3>(MPI_COMM_WORLD, pcout, max_err, n_skipped);

      run_both_overlap_modes<2, 1, 2>(MPI_COMM_WORLD, pcout, max_err, n_skipped);
      run_both_overlap_modes<2, 2, 2>(MPI_COMM_WORLD, pcout, max_err, n_skipped);
      run_both_overlap_modes<2, 3, 2>(MPI_COMM_WORLD, pcout, max_err, n_skipped);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-12;

  if (n_skipped > 0)
    pcout << "Note: " << n_skipped
          << " overlap configuration(s) skipped (build without device-aware MPI)." << std::endl;

  if (max_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_err << " (tolerance " << tolerance << ")"
        << std::endl;
  return 0;
}
