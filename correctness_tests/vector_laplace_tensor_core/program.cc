// Correctness check for Portable::VectorLaplaceOperator::vmult_tensor_core()
// / BK4::Parallel::TensorCore::f64_m8n8k4_mma() (bk4_cuda_kernels.cuh): the
// FP64 Tensor Core (mma.sync.aligned.m8n8k4.f64, needs compute capability
// >= 8.0) path, compared against vmult_dealii() (real deal.II's own
// Portable::FEEvaluation, taken as ground truth) and vmult_bk4() (the
// already cross-checked custom Kokkos kernel -- see tests/
// vector_laplace_bk4/program.cc) on a random vector.
//
// This only exercises anything when actually compiled by nvcc against a
// CUDA-enabled Kokkos build (MemorySpace::Default::kokkos_space needs to be
// Kokkos::CudaSpace) -- see bk4_cuda_kernels.cuh's __CUDACC__ guard. On a
// Serial/OpenMP-only Kokkos build (such as the one this project's own dev
// box currently has -- not enough memory there to build a CUDA-enabled
// deal.II) the tensor-core comparison is skipped and only the vmult_bk4()
// vs. vmult_dealii() sanity check runs, so this program still compiles and
// gives a meaningful (if partial) result anywhere.
//
// *** Status ***: PASSING on a real CUDA-enabled deal.II build (cluster),
// all three configs at machine precision, after fixing a real bug in
// bk4_cuda_kernels.cuh's Phase 3 (qr/qs/qt were paired with the wrong
// Grr/Grs/.../Gtt terms -- an unconditional R/T swap, wrong even for an
// isotropic G). Still open: a "malloc_consolidate(): unaligned fastbin
// chunk detected" crash during process teardown, after this program has
// already printed PASSED/FAILED and returned -- identical before and after
// the Phase 3 fix, so unrelated to it; likely an MPI/Kokkos/CUDA-runtime
// finalize-order issue or a deal.II-vs-this-CUDA-version compatibility
// issue rather than anything in this kernel, but unconfirmed.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>
#include <random>

#include "operators/portable_vector_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;

  constexpr int dim = 3; // bk4_cuda_kernels.cuh only implements dim == 3

  // Runs the comparison for one (fe_degree, n_components) combination on a
  // small hyper-cube mesh. vmult_tensor_core() derives its own
  // nelmtPerBatch internally from fe_degree (see tensor_core_nelmt_per_batch
  // in portable_vector_laplace_operator.h) -- no need to pick one here.
  // Returns the max relative error over whichever checks actually ran
  // (rel_err(bk4 vs dealii) always; rel_err(tensor_core vs dealii) only
  // when built with nvcc against a CUDA-enabled Kokkos) -- all of which
  // should be at machine precision.
  template <int fe_degree, int n_components>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_components = " << n_components << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(3);

    const FE_Q<dim> scalar_fe(fe_degree);
    FESystem<dim>   fe(scalar_fe, n_components);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero(n_components);
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);
    constraints.close();

    Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double> op(
      dof_handler, constraints, /* overlap_communication_computation = */ false);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src, dst_dealii, dst_bk4;
    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst_dealii);
    op.initialize_dof_vector(dst_bk4);

    {
      std::mt19937                           gen(42);
      std::uniform_real_distribution<double> dist(-1., 1.);

      LinearAlgebra::ReadWriteVector<double> rw(locally_owned_dofs);
      for (const auto idx : locally_owned_dofs)
        rw(idx) = dist(gen);
      src.import_elements(rw, VectorOperation::insert);
    }

    // vmult_dealii() reads src at constrained DoFs directly (it relies on
    // copy_constrained_values() afterwards), while vmult_bk4()/
    // vmult_tensor_core() never read constrained entries at all -- zero
    // them up front so every variant sees the same effective input, exactly
    // as tests/vector_laplace_bk4/program.cc and abstracted_batched_laplace/
    // program.cc do.
    op.get_matrix_free().set_constrained_values(0., src);

    op.vmult_dealii(dst_dealii, src);
    op.vmult_bk4(dst_bk4, src);

    op.get_matrix_free().set_constrained_values(0., dst_dealii);
    op.get_matrix_free().set_constrained_values(0., dst_bk4);

    const double norm_dealii = dst_dealii.l2_norm();

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = dst_bk4;
    diff -= dst_dealii;
    const double rel_err_bk4 = norm_dealii > 0 ? diff.l2_norm() / norm_dealii : diff.l2_norm();

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_dealii| = " << norm_dealii
          << ", rel_err(bk4 vs dealii) = " << rel_err_bk4 << std::endl;

    double max_rel_err = rel_err_bk4;

#ifdef __CUDACC__
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dst_tensor_core;
    op.initialize_dof_vector(dst_tensor_core);

    op.vmult_tensor_core(dst_tensor_core, src);
    op.get_matrix_free().set_constrained_values(0., dst_tensor_core);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff_tc = dst_tensor_core;
    diff_tc -= dst_dealii;
    const double rel_err_tc = norm_dealii > 0 ? diff_tc.l2_norm() / norm_dealii : diff_tc.l2_norm();

    pcout << "  rel_err(tensor_core vs dealii) = " << rel_err_tc << std::endl;

    max_rel_err = std::max(max_rel_err, rel_err_tc);
#else
    pcout << "  rel_err(tensor_core vs dealii) = SKIPPED "
             "(this translation unit was not compiled by nvcc)"
          << std::endl;
#endif

    // Sanity-check compute_diagonal(): a Laplacian's diagonal must be
    // strictly positive everywhere (compute_diagonal() itself Asserts this
    // in debug mode already -- this just makes the check visible here too).
    op.compute_diagonal();
    const auto &inverse_diagonal = op.get_matrix_diagonal_inverse()->get_vector();
    LinearAlgebra::ReadWriteVector<double> diagonal_host(locally_owned_dofs);
    diagonal_host.import_elements(inverse_diagonal, VectorOperation::insert);
    double min_inverse_diagonal = std::numeric_limits<double>::max();
    for (const auto idx : locally_owned_dofs)
      min_inverse_diagonal = std::min(min_inverse_diagonal, diagonal_host(idx));
    pcout << "  min(1/diagonal) = " << min_inverse_diagonal << std::endl << std::endl;

    return max_rel_err;
  }
} // namespace multigrid

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

#ifdef __CUDACC__
  pcout << "Built with nvcc: vmult_tensor_core() will actually run on the GPU." << std::endl
        << std::endl;
#else
  pcout << "NOT built with nvcc: vmult_tensor_core() checks below are skipped -- "
           "rebuild this target with a CUDA-enabled Kokkos to exercise them."
        << std::endl
        << std::endl;
#endif

  double max_rel_err = 0.;

  try
    {
      max_rel_err = std::max(max_rel_err, run_test<2, 3>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<4, 2>(MPI_COMM_WORLD, pcout));
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-10;

  if (max_rel_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_rel_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_rel_err << " (tolerance " << tolerance << ")"
        << std::endl;

  return 0;
}
