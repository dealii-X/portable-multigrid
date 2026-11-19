// Correctness check for Portable::LaplaceOperatorBK3 (portable_laplace_operator_bk3.h)
// on a curved, sheared, non-axis-aligned mesh: vmult(), Tvmult() and
// compute_diagonal() are compared against a CPU FEValues reference built with
// the same MappingQ(fe_degree) the operator uses internally.
//
// Every other test in correctness_tests/ runs on an axis-aligned hyper-cube,
// where the inverse Jacobian is diagonal and constant, so a wrong contraction
// index in the G-tensor (J^{-1} J^{-T} vs J^{-T} J^{-1}) or a wrong Jacobian
// layout would go unnoticed. Here all dim*(dim+1)/2 entries of G vary per
// quadrature point.
//
// The reference does exactly what the operator does with Dirichlet
// constraints: constrained source entries are read as zero, constrained rows
// of the result are not accumulated, and afterwards dst[c] = src[c] on every
// constrained DoF (copy_constrained_values). The diagonal is expected to be 1
// on constrained DoFs.
//
// Runs for fe_degree = 1..4 in double (tolerance 1e-12) and float (1e-5),
// dim = 2 and 3, with and without overlap_communication_computation. The
// overlap configuration is skipped (not failed) when the build cannot
// support it, see run_test().

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <string>

#include "deformed_mesh_test_utils.h"
#include "operators/portable_laplace_operator_bk3.h"

namespace multigrid
{
  using namespace dealii;
  using namespace deformed_mesh_test;

  // Returns the largest relative error over vmult, Tvmult and the diagonal
  // for one configuration, or -1 if the configuration was skipped.
  template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout, const bool overlap)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << ", "
          << (std::is_same_v<Number, double> ? "double" : "float")
          << ", overlap = " << (overlap ? "true" : "false") << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
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

    const auto op =
      make_operator_or_skip<Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d>>(
        pcout, overlap, dof_handler, constraints, overlap);
    if (!op)
      return -1.;

    AssertThrow(op->m() == dof_handler.n_dofs() && op->n() == dof_handler.n_dofs(),
                ExcMessage("m()/n() do not match the number of DoFs."));

    using VectorType = DeviceVectorType<Number>;
    VectorType src, dst_gpu, dst_tvmult_gpu, dst_ref, diag_ref;
    op->initialize_dof_vector(src);
    op->initialize_dof_vector(dst_gpu);
    op->initialize_dof_vector(dst_tvmult_gpu);
    op->initialize_dof_vector(dst_ref);
    op->initialize_dof_vector(diag_ref);

    // deliberately nonzero on Dirichlet DoFs so copy_constrained_values() is
    // exercised
    fill_source(src, locally_owned_dofs);

    op->vmult(dst_gpu, src);
    op->Tvmult(dst_tvmult_gpu, src);
    op->compute_diagonal();
    Kokkos::fence();

    reference_laplace<dim, Number>(mapping,
                                   dof_handler,
                                   constraints,
                                   n_q_points_1d,
                                   locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator,
                                   src,
                                   dst_ref,
                                   diag_ref);

    const double err_vmult  = relative_l2_difference(dst_gpu, dst_ref);
    const double err_tvmult = relative_l2_difference(dst_tvmult_gpu, dst_ref);

    const double err_diag =
      max_relative_diagonal_error(op->get_matrix_diagonal_inverse()->get_vector(),
                                  diag_ref,
                                  locally_owned_dofs,
                                  mpi_communicator);

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_ref| = " << dst_ref.l2_norm()
          << ", rel_err(vmult) = " << err_vmult << ", rel_err(Tvmult) = " << err_tvmult
          << ", max_rel_err(diag) = " << err_diag << std::endl
          << std::endl;

    return std::max({err_vmult, err_tvmult, err_diag});
  }



  template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
  void
  run_both_overlap_modes(const MPI_Comm      mpi_communicator,
                         ConditionalOStream &pcout,
                         double             &max_err,
                         unsigned int       &n_skipped)
  {
    for (const bool overlap : {false, true})
      {
        const double err =
          run_test<dim, fe_degree, Number, n_q_points_1d>(mpi_communicator, pcout, overlap);
        if (err < 0.)
          ++n_skipped;
        else
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

  double       max_err_fp64 = 0., max_err_fp32 = 0.;
  unsigned int n_skipped = 0;

  try
    {
      // 3D, fp64, default quadrature (nq == p + 1)
      run_both_overlap_modes<3, 1, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 2, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 3, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 4, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);

      // 3D, fp64, over-integrated (nq == p + 2), what tests/bp3 uses
      run_both_overlap_modes<3, 2, double, 4>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 3, double, 5>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);

      // 3D, fp32 (nq == p + 1 is odd for even p)
      run_both_overlap_modes<3, 1, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 2, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 3, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 4, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);

      // 2D, fp64
      run_both_overlap_modes<2, 1, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 2, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 3, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 4, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance_fp64 = 1e-12;
  const double tolerance_fp32 = 1e-5;

  if (n_skipped > 0)
    pcout << "Note: " << n_skipped
          << " overlap configuration(s) skipped (build without device-aware MPI)." << std::endl;

  if (max_err_fp64 > tolerance_fp64 || max_err_fp32 > tolerance_fp32)
    {
      pcout << "FAILED: max relative error fp64 " << max_err_fp64 << " (tolerance "
            << tolerance_fp64 << "), fp32 " << max_err_fp32 << " (tolerance " << tolerance_fp32
            << ")" << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error fp64 " << max_err_fp64 << " (tolerance " << tolerance_fp64
        << "), fp32 " << max_err_fp32 << " (tolerance " << tolerance_fp32 << ")" << std::endl;

  return 0;
}
