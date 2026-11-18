// Correctness check for Portable::LaplaceOperatorBK3's new n_q_points_1d
// template parameter and vmult_tensor_core() (portable_laplace_operator_bk3.h):
// compares vmult() (the custom BK3 Kokkos kernel) against
// Portable::LaplaceOperator::vmult_dealii() (real deal.II's own
// Portable::FEEvaluation path, taken as ground truth) on a random vector, at
// both the operator's default n_q_points_1d (nq == nm) and an
// over-integrated one (nq == nm + 1, what tests/bp3/program.cc uses) --
// mirrors tests/vector_laplace_overintegrated/program.cc's role for
// VectorLaplaceOperator.
//
// Also force-instantiates vmult_tensor_core() (dead code when this
// translation unit isn't compiled by nvcc, exactly like tests/
// vector_laplace_bk4/program.cc does for VectorLaplaceOperator) purely to
// catch compile errors in its non-CUDA fallback branch -- it reuses
// bk4_cuda_kernels.cuh's vector-Laplacian tensor-core kernel with
// n_components == 1, which is new wiring this session and has no other
// compile-time check anywhere else in this project.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>
#include <random>

#include "operators/portable_laplace_operator.h"
#include "operators/portable_laplace_operator_bk3.h"

namespace multigrid
{
  using namespace dealii;

  // Runs the comparison for one (dim, fe_degree, n_q_points_1d) combination
  // on a small hyper-cube mesh. Returns the relative error between
  // LaplaceOperatorBK3::vmult() and LaplaceOperator::vmult_dealii(), which
  // should be at machine precision.
  template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);

    const FE_Q<dim> fe(fe_degree);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero;
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    constraints.close();

    // Two independently-constructed real deal.II Portable::MatrixFree
    // instances (one inside each operator) reinit()'d against the *same*
    // DoFHandler don't stay resolvable through deal.II's own registry-
    // based distributed_cell_loop() (it tracks the *latest* reinit() per
    // DoFHandler by partitioner identity) -- so device vectors can't be
    // shared/compared directly across op_dealii and op_bk3 the way the
    // existing single-operator tests (tests/vector_laplace_bk4/, etc.) do.
    // Round-tripping through a host-side ReadWriteVector (keyed by
    // locally_owned_dofs, not by any MatrixFree's internal partitioner
    // identity) sidesteps that entirely.
    Portable::LaplaceOperator<dim, fe_degree, double> op_dealii(
      dof_handler, constraints, /* overlap_communication_computation = */ false);
    Portable::LaplaceOperatorBK3<dim, fe_degree, double, n_q_points_1d> op_bk3(
      dof_handler, constraints, /* overlap_communication_computation = */ false);

    LinearAlgebra::ReadWriteVector<double> rw_src(locally_owned_dofs);
    {
      std::mt19937                           gen(42);
      std::uniform_real_distribution<double> dist(-1., 1.);

      for (const auto idx : locally_owned_dofs)
        rw_src(idx) = dist(gen);
    }

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src_dealii, dst_dealii;
    op_dealii.initialize_dof_vector(src_dealii);
    op_dealii.initialize_dof_vector(dst_dealii);
    src_dealii.import_elements(rw_src, VectorOperation::insert);

    // vmult_dealii() reads src at constrained DoFs directly (it relies on
    // copy_constrained_values() afterwards), while vmult() never reads
    // constrained entries at all -- zero them in the host copy up front
    // (before importing into either operator's own device vector) so both
    // variants see the same effective input, exactly as tests/
    // vector_laplace_bk4/program.cc does for a single operator.
    op_dealii.get_matrix_free().set_constrained_values(0., src_dealii);
    op_dealii.vmult_dealii(dst_dealii, src_dealii);
    op_dealii.get_matrix_free().set_constrained_values(0., dst_dealii);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src_bk3, dst_bk3;
    op_bk3.initialize_dof_vector(src_bk3);
    op_bk3.initialize_dof_vector(dst_bk3);
    src_bk3.import_elements(rw_src, VectorOperation::insert);
    op_bk3.set_constrained_values(0., src_bk3);
    op_bk3.vmult(dst_bk3, src_bk3);
    op_bk3.set_constrained_values(0., dst_bk3);

    // vmult_tensor_core() is dim == 3 + double only and needs an actual
    // nvcc/CUDA build to do anything -- this sandbox has neither a CUDA
    // compiler nor a CUDA-enabled Kokkos build, so this is dead code
    // (never executed) purely to force the compiler to instantiate and
    // typecheck vmult_tensor_core()'s non-CUDA fallback branch (the
    // Assert(false, ...) path) as part of this compile check.
    if constexpr (dim == 3)
      if (false)
        op_bk3.vmult_tensor_core(dst_bk3, src_bk3);

    LinearAlgebra::ReadWriteVector<double> rw_dealii(locally_owned_dofs), rw_bk3(locally_owned_dofs);
    rw_dealii.import_elements(dst_dealii, VectorOperation::insert);
    rw_bk3.import_elements(dst_bk3, VectorOperation::insert);

    double norm_dealii_sq = 0., diff_sq = 0.;
    for (const auto idx : locally_owned_dofs)
      {
        norm_dealii_sq += rw_dealii(idx) * rw_dealii(idx);
        const double d = rw_bk3(idx) - rw_dealii(idx);
        diff_sq += d * d;
      }
    norm_dealii_sq = Utilities::MPI::sum(norm_dealii_sq, mpi_communicator);
    diff_sq        = Utilities::MPI::sum(diff_sq, mpi_communicator);

    const double norm_dealii = std::sqrt(norm_dealii_sq);
    const double rel_err = norm_dealii > 0 ? std::sqrt(diff_sq) / norm_dealii : std::sqrt(diff_sq);

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_dealii| = " << norm_dealii
          << ", rel_err(bk3 vs dealii) = " << rel_err << std::endl
          << std::endl;

    return rel_err;
  }
} // namespace multigrid

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double max_rel_err = 0.;

  try
    {
      // Default n_q_points_1d (nq == nm), regression check.
      max_rel_err = std::max(max_rel_err, run_test<2, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 3>(MPI_COMM_WORLD, pcout));

      // Over-integrated (nq == nm + 1), what tests/bp3/program.cc uses.
      max_rel_err =
        std::max(max_rel_err, run_test<3, 2, /* n_q_points_1d = */ 4>(MPI_COMM_WORLD, pcout));
      max_rel_err =
        std::max(max_rel_err, run_test<3, 3, /* n_q_points_1d = */ 5>(MPI_COMM_WORLD, pcout));
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
