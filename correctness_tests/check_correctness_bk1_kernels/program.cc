// Correctness check for the "Abstracted" BK1 transfer kernels
// (include/kernels/bk1_kokkos_kernels.h) -- KokkosProlongationBatchedKernelAbstracted()/
// KokkosRestrictionBatchedKernelAbstracted(), built from
// Custom::Parallel::EvaluatorTensorProduct instead of the hand-unrolled
// per-direction loops in KokkosProlongationBatchedKernel()/
// KokkosRestrictionBatchedKernel() -- against those original, already-used-
// in-production kernels.
//
// Two independent checks, both comparing prolongate_and_add()/
// prolongate_and_add_new() and restrict_and_add()/restrict_and_add_new()
// (base/portable_mg_transfer_base.h) on the same random input:
//
//  1. run_test(): a real Portable::PolynomialTransfer on two DoFHandlers
//     (same triangulation, degrees p_coarse/p_fine) --
//     include/multigrid/portable_polynomial_transfer.h.
//
//  2. run_geometric_test(): a real Portable::GeometricTransfer on two
//     adjacent levels of a geometric coarsening sequence (same degree,
//     different mesh) -- include/multigrid/portable_geometric_transfer.h.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

// See correctness_tests/check_correctness_laplace_operator_batched/program.cc
// for why these are needed directly here.
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <cmath>
#include <iostream>
#include <random>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_laplace_operator_bk3.h"

using namespace dealii;

using Number = double;

// Shared by both run_test()/run_geometric_test() below.
bool
compare_vectors(const char                                                             *label,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &orig,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &fresh)
{
  LinearAlgebra::ReadWriteVector<Number> rw_orig(orig.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_fresh(fresh.locally_owned_elements());
  rw_orig.import_elements(orig, VectorOperation::insert);
  rw_fresh.import_elements(fresh, VectorOperation::insert);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (const auto idx : orig.locally_owned_elements())
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(rw_orig(idx) - rw_fresh(idx)));
      max_abs_val  = std::max(max_abs_val, std::abs(rw_orig(idx)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "  " << label << ": max |orig| = " << max_abs_val
            << "   max |orig - new| = " << max_abs_diff << "   " << (pass ? "PASS" : "FAIL")
            << std::endl;

  return pass;
}

template <int dim, int p_coarse, int p_fine>
bool
run_test()
{
  const FE_Q<dim> fe_coarse(p_coarse);
  const FE_Q<dim> fe_fine(p_fine);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);

  DoFHandler<dim> dof_handler_coarse(triangulation);
  DoFHandler<dim> dof_handler_fine(triangulation);
  dof_handler_coarse.distribute_dofs(fe_coarse);
  dof_handler_fine.distribute_dofs(fe_fine);

  AffineConstraints<Number> constraints_coarse, constraints_fine;
  DoFTools::make_hanging_node_constraints(dof_handler_coarse, constraints_coarse);
  constraints_coarse.close();
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraints_fine);
  constraints_fine.close();

  Portable::LaplaceOperatorBK3<dim, p_coarse, Number> op_coarse(dof_handler_coarse,
                                                                constraints_coarse,
                                                                /*overlap_communication_computation=*/
                                                                false);
  Portable::LaplaceOperatorBK3<dim, p_fine, Number> op_fine(dof_handler_fine,
                                                             constraints_fine,
                                                             /*overlap_communication_computation=*/
                                                             false);

  Portable::PolynomialTransfer<dim, p_coarse, p_fine, Number> transfer;
  transfer.reinit(op_coarse.get_matrix_free(),
                  op_fine.get_matrix_free(),
                  constraints_coarse,
                  constraints_fine);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src_coarse, src_fine;
  op_coarse.initialize_dof_vector(src_coarse);
  op_fine.initialize_dof_vector(src_fine);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw_coarse(src_coarse.locally_owned_elements());
    for (const auto idx : src_coarse.locally_owned_elements())
      rw_coarse(idx) = dist(gen);
    src_coarse.import_elements(rw_coarse, VectorOperation::insert);

    LinearAlgebra::ReadWriteVector<Number> rw_fine(src_fine.locally_owned_elements());
    for (const auto idx : src_fine.locally_owned_elements())
      rw_fine(idx) = dist(gen);
    src_fine.import_elements(rw_fine, VectorOperation::insert);
  }

  bool all_passed = true;

  std::cout << "dim = " << dim << ", p_coarse = " << p_coarse << ", p_fine = " << p_fine
            << ", n_dofs_coarse = " << dof_handler_coarse.n_dofs()
            << ", n_dofs_fine = " << dof_handler_fine.n_dofs() << std::endl;

  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst_orig, dst_new;
    op_fine.initialize_dof_vector(dst_orig);
    op_fine.initialize_dof_vector(dst_new);
    dst_orig = 0.;
    dst_new  = 0.;

    transfer.prolongate_and_add(dst_orig, src_coarse);
    transfer.prolongate_and_add_new(dst_new, src_coarse);

    all_passed &= compare_vectors("prolongate_and_add", dst_orig, dst_new);
  }

  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst_orig, dst_new;
    op_coarse.initialize_dof_vector(dst_orig);
    op_coarse.initialize_dof_vector(dst_new);
    dst_orig = 0.;
    dst_new  = 0.;

    transfer.restrict_and_add(dst_orig, src_fine);
    transfer.restrict_and_add_new(dst_new, src_fine);

    all_passed &= compare_vectors("restrict_and_add", dst_orig, dst_new);
  }

  return all_passed;
}

// Portable::GeometricTransfer analog of run_test() above -- same degree on
// both sides, two adjacent levels of a geometric coarsening sequence
// instead of two DoFHandlers at different polynomial degree on the same
// mesh.
template <int dim, int fe_degree>
bool
run_geometric_test()
{
  const FE_Q<dim> fe(fe_degree);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(3);

  const auto coarsening_sequence =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      triangulation, RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(4));

  if (coarsening_sequence.size() < 2)
    {
      std::cout << "dim = " << dim << ", fe_degree = " << fe_degree
                << ": coarsening sequence too short (" << coarsening_sequence.size()
                << " levels), skipping" << std::endl;
      return true;
    }

  const Triangulation<dim> &tria_coarse = *coarsening_sequence[coarsening_sequence.size() - 2];
  const Triangulation<dim> &tria_fine   = *coarsening_sequence[coarsening_sequence.size() - 1];

  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  DoFHandler<dim> dof_handler_fine(tria_fine);
  dof_handler_coarse.distribute_dofs(fe);
  dof_handler_fine.distribute_dofs(fe);

  AffineConstraints<Number> constraints_coarse, constraints_fine;
  DoFTools::make_hanging_node_constraints(dof_handler_coarse, constraints_coarse);
  constraints_coarse.close();
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraints_fine);
  constraints_fine.close();

  Portable::LaplaceOperatorBK3<dim, fe_degree, Number> op_coarse(dof_handler_coarse,
                                                                 constraints_coarse,
                                                                 /*overlap_communication_computation=*/
                                                                 false);
  Portable::LaplaceOperatorBK3<dim, fe_degree, Number> op_fine(dof_handler_fine,
                                                               constraints_fine,
                                                               /*overlap_communication_computation=*/
                                                               false);

  Portable::GeometricTransfer<dim, fe_degree, Number> transfer;
  transfer.reinit(op_coarse.get_matrix_free(),
                  op_fine.get_matrix_free(),
                  constraints_coarse,
                  constraints_fine);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src_coarse, src_fine;
  op_coarse.initialize_dof_vector(src_coarse);
  op_fine.initialize_dof_vector(src_fine);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw_coarse(src_coarse.locally_owned_elements());
    for (const auto idx : src_coarse.locally_owned_elements())
      rw_coarse(idx) = dist(gen);
    src_coarse.import_elements(rw_coarse, VectorOperation::insert);

    LinearAlgebra::ReadWriteVector<Number> rw_fine(src_fine.locally_owned_elements());
    for (const auto idx : src_fine.locally_owned_elements())
      rw_fine(idx) = dist(gen);
    src_fine.import_elements(rw_fine, VectorOperation::insert);
  }

  bool all_passed = true;

  std::cout << "[geometric] dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_dofs_coarse = " << dof_handler_coarse.n_dofs()
            << ", n_dofs_fine = " << dof_handler_fine.n_dofs() << std::endl;

  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst_orig, dst_new;
    op_fine.initialize_dof_vector(dst_orig);
    op_fine.initialize_dof_vector(dst_new);
    dst_orig = 0.;
    dst_new  = 0.;

    transfer.prolongate_and_add(dst_orig, src_coarse);
    transfer.prolongate_and_add_new(dst_new, src_coarse);

    all_passed &= compare_vectors("prolongate_and_add", dst_orig, dst_new);
  }

  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst_orig, dst_new;
    op_coarse.initialize_dof_vector(dst_orig);
    op_coarse.initialize_dof_vector(dst_new);
    dst_orig = 0.;
    dst_new  = 0.;

    transfer.restrict_and_add(dst_orig, src_fine);
    transfer.restrict_and_add_new(dst_new, src_fine);

    all_passed &= compare_vectors("restrict_and_add", dst_orig, dst_new);
  }

  return all_passed;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  bool all_passed = true;
  all_passed &= run_test<3, 1, 2>();
  all_passed &= run_test<3, 1, 3>();
  all_passed &= run_test<3, 2, 4>();
  all_passed &= run_test<2, 1, 2>();
  all_passed &= run_test<2, 1, 4>();

  all_passed &= run_geometric_test<3, 1>();
  all_passed &= run_geometric_test<3, 2>();
  all_passed &= run_geometric_test<2, 1>();
  all_passed &= run_geometric_test<2, 3>();

  std::cout << (all_passed ? "ALL PASS" : "SOME FAILED") << std::endl;

  return all_passed ? 0 : 1;
}
