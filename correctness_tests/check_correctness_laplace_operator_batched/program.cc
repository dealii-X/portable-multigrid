// Correctness check for Portable::LaplaceOperator::vmult_new()
// (cell_loop_batched(), include/operators/portable_laplace_operator.h)
// against the operator's existing, known-correct vmult() (cell_loop()) --
// both computed from the exact same Portable::LaplaceOperator instance
// (same DoFHandler/AffineConstraints/MatrixFree setup), applied to the same
// random right-hand side, for several (dim, fe_degree) combinations.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

// Pulls in Portable::internal::EvaluatorTensorProduct/EvaluatorVariant and
// Portable::FEEvaluation, both used (without including their declaring
// header directly) by kernels/portable_local_laplace_operator.h and
// operators/portable_laplace_operator_quad.h respectively -- every other
// translation unit in this codebase happens to pull this header in first
// via something else, so this gap has never surfaced before.
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <random>

#include "operators/portable_laplace_operator.h"

using namespace dealii;

using Number = double;

template <int dim, int fe_degree>
bool
run_test(const bool overlap_communication_computation = false)
{
  const FE_Q<dim> fe(fe_degree);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
    dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  Portable::LaplaceOperator<dim, fe_degree, Number> laplace_operator(
    dof_handler, constraints, overlap_communication_computation);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src, dst_orig, dst_batched;
  laplace_operator.initialize_dof_vector(src);
  laplace_operator.initialize_dof_vector(dst_orig);
  laplace_operator.initialize_dof_vector(dst_batched);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw(src.locally_owned_elements());
    for (const auto idx : src.locally_owned_elements())
      rw(idx) = dist(gen);
    src.import_elements(rw, VectorOperation::insert);
  }

  laplace_operator.vmult(dst_orig, src);
  laplace_operator.vmult_new(dst_batched, src);

  LinearAlgebra::ReadWriteVector<Number> rw_orig(dst_orig.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_batched(dst_batched.locally_owned_elements());
  rw_orig.import_elements(dst_orig, VectorOperation::insert);
  rw_batched.import_elements(dst_batched, VectorOperation::insert);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (const auto idx : dst_orig.locally_owned_elements())
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(rw_orig(idx) - rw_batched(idx)));
      max_abs_val  = std::max(max_abs_val, std::abs(rw_orig(idx)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_dofs = " << dof_handler.n_dofs()
            << ", overlap_communication_computation = " << overlap_communication_computation
            << std::endl;
  std::cout << "  max |vmult|              = " << max_abs_val << std::endl;
  std::cout << "  max |vmult - vmult_new|  = " << max_abs_diff << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  bool all_passed = true;
  all_passed &= run_test<3, 3>();
  all_passed &= run_test<3, 1>();
  all_passed &= run_test<3, 2>();
  all_passed &= run_test<2, 4>();
  all_passed &= run_test<2, 1>();
  all_passed &= run_test<2, 3>();

  // overlap_communication_computation == true (cell_loop_batched()'s 3-color
  // update_ghost_values_start/finish + compress_start/finish path) requires
  // device-aware MPI and can't be exercised on this CPU-only environment --
  // left as a runtime-selectable parameter to run_test() for when it can be
  // (e.g. on JUPITER).

  std::cout << (all_passed ? "ALL PASS" : "SOME FAILED") << std::endl;

  return all_passed ? 0 : 1;
}
