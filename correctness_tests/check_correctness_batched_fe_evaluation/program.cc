// Correctness check for Custom::Parallel::FEEvaluation
// (include/kernels/portable_batched_fe_evaluation.h) -- the batched,
// deal.II-Portable::FEEvaluation-faithful get_value()/get_gradient()/
// submit_value()/submit_gradient() accessor API, meant to be called from a
// per-quadrature-point Functor exactly like step-64's HelmholtzOperatorQuad.
//
// Two independent checks:
//
//  1. run_test(): builds a real Portable::LaplaceOperator on a Dirichlet-only
//     (globally, not adaptively, refined -- no hanging nodes) problem and
//     compares its ground-truth vmult_bk3() (best-performing, fully-abstracted
//     BK3 kernel) against three other entry points (operators/
//     portable_laplace_operator.h):
//       - vmult_dealii(): real deal.II's own Portable::FEEvaluation/
//         MatrixFree::cell_loop() (LocalLaplaceOperatorStep64, kernels/
//         portable_local_laplace_operator.h) -- unlike every kernel this
//         project writes itself, this one does NOT mask constrained DoFs
//         internally (real deal.II only auto-masks hanging-node
//         constraints, not Dirichlet ones -- the reverse of this project's
//         own kernels, which mask Dirichlet but don't support hanging nodes
//         yet; Ivan's words: "it's on my long todo list to enable
//         [Dirichlet masking] in deal.II as well"). Made comparable by
//         zeroing src's (and, belt-and-suspenders, every dst's) constrained
//         entries via MatrixFree::set_constrained_values() before comparing
//         -- see below.
//       - vmult_dealii_batched(): LocalLaplaceOperatorGeneric (kernels/
//         portable_local_laplace_operator_batched.h) -- the standard,
//         step-64-style pattern (read_dof_values()/evaluate(gradients)/
//         get_gradient()/submit_gradient()/integrate(gradients)/
//         distribute_local_to_global()), batched via cell_loop_batched().
//       - vmult_dealii_batched_fused(): LocalLaplaceOperatorGenericSplit,
//         same math via the split evaluate_values()/evaluate_gradients()/
//         integrate_gradients()/integrate_values() instead of the combined,
//         EvaluationFlags-driven evaluate()/integrate().
//     None of these three use LocalLaplaceOperatorBatched's G_tensor-fused
//     path (that's vmult_bk3()) -- all three go through FEEvaluation's
//     on-the-fly inv_jacobian/JxW multiply instead. Compares all four dst
//     vectors on random (Dirichlet-zeroed) input.
//
//  2. run_integrate_add_test(): isolates the new `add` template parameter
//     on FEEvaluationImplTransformToCollocation::integrate_gradients()
//     (portable_evaluation_kernels.h) -- the fusion FEEvaluation::integrate()
//     uses when both values and gradients were submitted -- and checks
//     integrate_gradients<true>(..., values) == values_before +
//     integrate_gradients<false>(..., values), directly, with no MatrixFree
//     setup needed. Not exercised by run_test() above, since pure Laplace
//     stiffness never calls submit_value().

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

// See correctness_tests/check_correctness_laplace_operator_batched/program.cc
// for why these are needed directly here.
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <random>

#include "matrix_free/portable_batched_fe_evaluation.h"
#include "operators/portable_laplace_operator.h"

using namespace dealii;

using Number = double;

template <int dim, int fe_degree>
bool
run_test()
{
  const FE_Q<dim> fe(fe_degree);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // No make_hanging_node_constraints() -- the mesh below is only globally
  // (not adaptively) refined, so there are no hanging nodes yet, and this
  // project's own kernels (BK3, batched FEEvaluation) only mask Dirichlet
  // boundary constraints so far, not hanging-node ones (the reverse of what
  // real deal.II's own Portable::MatrixFree/FEEvaluation does internally --
  // hanging nodes yes, Dirichlet no). Keep this test Dirichlet-only until
  // that's reconciled.
  AffineConstraints<Number> constraints;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  Portable::LaplaceOperator<dim, fe_degree, Number> laplace_operator(
    dof_handler, constraints, /*overlap_communication_computation=*/false);

  // -- random input, apply all paths, compare --
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src, dst_bk3, dst_dealii,
    dst_generic, dst_generic_split;
  laplace_operator.initialize_dof_vector(src);
  laplace_operator.initialize_dof_vector(dst_bk3);
  laplace_operator.initialize_dof_vector(dst_dealii);
  laplace_operator.initialize_dof_vector(dst_generic);
  laplace_operator.initialize_dof_vector(dst_generic_split);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw(src.locally_owned_elements());
    for (const auto idx : src.locally_owned_elements())
      rw(idx) = dist(gen);
    src.import_elements(rw, VectorOperation::insert);
  }

  // vmult_dealii() drives real deal.II's own Portable::FEEvaluation/
  // MatrixFree::cell_loop(), which -- unlike this project's own masked
  // kernels -- reads the *actual* src value at Dirichlet-constrained DoFs
  // rather than always treating it as zero. Zero those entries in src
  // up front so every vmult variant below sees the same (homogeneous
  // Dirichlet) input; this project's own kernels are invariant to this
  // already (they never read src at constrained DoFs at all), so this only
  // changes what vmult_dealii() sees.
  laplace_operator.get_matrix_free().set_constrained_values(0., src);

  // vmult_bk3() is ground truth here (see file-header comment above) -- not
  // vmult().
  laplace_operator.vmult_bk3(dst_bk3, src);
  laplace_operator.vmult_dealii(dst_dealii, src);

  // vmult_dealii_batched()/vmult_dealii_batched_fused() (operators/
  // portable_laplace_operator.h) drive LocalLaplaceOperatorGeneric/
  // LocalLaplaceOperatorGenericSplit through cell_loop_batched() internally
  // -- exercise those production entry points directly, no manual
  // dof_indices_per_color/color-loop replication needed.
  laplace_operator.vmult_dealii_batched(dst_generic, src);
  laplace_operator.vmult_dealii_batched_fused(dst_generic_split, src);

  // Belt-and-suspenders: also zero the constrained entries of every dst
  // before comparing (each vmult already leaves them at 0 given the
  // zeroed src above, via either copy_constrained_values() (vmult_dealii())
  // or simply never writing them (the masked kernels) -- this just makes
  // that explicit rather than relying on it).
  laplace_operator.get_matrix_free().set_constrained_values(0., dst_bk3);
  laplace_operator.get_matrix_free().set_constrained_values(0., dst_dealii);
  laplace_operator.get_matrix_free().set_constrained_values(0., dst_generic);
  laplace_operator.get_matrix_free().set_constrained_values(0., dst_generic_split);

  LinearAlgebra::ReadWriteVector<Number> rw_orig(dst_bk3.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_dealii(dst_dealii.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_generic(dst_generic.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_generic_split(dst_generic_split.locally_owned_elements());
  rw_orig.import_elements(dst_bk3, VectorOperation::insert);
  rw_dealii.import_elements(dst_dealii, VectorOperation::insert);
  rw_generic.import_elements(dst_generic, VectorOperation::insert);
  rw_generic_split.import_elements(dst_generic_split, VectorOperation::insert);

  double max_abs_diff        = 0.;
  double max_abs_diff_split  = 0.;
  double max_abs_diff_dealii = 0.;
  double max_abs_val         = 0.;
  for (const auto idx : dst_bk3.locally_owned_elements())
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(rw_orig(idx) - rw_generic(idx)));
      max_abs_diff_split =
        std::max(max_abs_diff_split, std::abs(rw_orig(idx) - rw_generic_split(idx)));
      max_abs_diff_dealii = std::max(max_abs_diff_dealii, std::abs(rw_orig(idx) - rw_dealii(idx)));
      max_abs_val         = std::max(max_abs_val, std::abs(rw_orig(idx)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val) &&
                    max_abs_diff_split < 1e-10 * std::max(1., max_abs_val) &&
                    max_abs_diff_dealii < 1e-10 * std::max(1., max_abs_val);

  std::cout << "dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_dofs = " << dof_handler.n_dofs() << std::endl;
  std::cout << "  max |vmult_bk3|                       = " << max_abs_val << std::endl;
  std::cout << "  max |vmult_bk3 - vmult_dealii|         = " << max_abs_diff_dealii << std::endl;
  std::cout << "  max |vmult_bk3 - fe_eval_based|        = " << max_abs_diff << std::endl;
  std::cout << "  max |vmult_bk3 - fe_eval_based_split|  = " << max_abs_diff_split << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}



template <int dim, int nq>
bool
run_integrate_add_test()
{
  constexpr int nq_total = Utilities::pow(nq, dim);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients(
    "co_shape_gradients", nq * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> gradients("gradients", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_preexisting(
    "values_preexisting", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_add("values_add", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_noadd("values_noadd", nq_total);

  {
    std::mt19937                           gen(7);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto gradients_host = Kokkos::create_mirror_view(gradients);
    for (int i = 0; i < dim * nq_total; ++i)
      gradients_host(i) = dist(gen);
    Kokkos::deep_copy(gradients, gradients_host);

    auto values_preexisting_host = Kokkos::create_mirror_view(values_preexisting);
    auto values_add_host         = Kokkos::create_mirror_view(values_add);
    for (int i = 0; i < nq_total; ++i)
      values_add_host(i) = values_preexisting_host(i) = dist(gen);
    Kokkos::deep_copy(values_preexisting, values_preexisting_host);
    Kokkos::deep_copy(values_add, values_add_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int threadIdx = team_member.team_rank();
      const int blockSize = team_member.team_size();
      const int batchIdx  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nq, nq, Number> fe_eval(
        team_member, nullptr, co_shape_gradients.data(), 1, 1, batchIdx, threadIdx, blockSize);

      fe_eval.template integrate_gradients<false>(gradients.data(), values_noadd.data());
      fe_eval.template integrate_gradients<true>(gradients.data(), values_add.data());
    });
  Kokkos::fence();

  auto values_preexisting_host = Kokkos::create_mirror_view(values_preexisting);
  auto values_add_host         = Kokkos::create_mirror_view(values_add);
  auto values_noadd_host       = Kokkos::create_mirror_view(values_noadd);
  Kokkos::deep_copy(values_preexisting_host, values_preexisting);
  Kokkos::deep_copy(values_add_host, values_add);
  Kokkos::deep_copy(values_noadd_host, values_noadd);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (int i = 0; i < nq_total; ++i)
    {
      const double expected = values_preexisting_host(i) + values_noadd_host(i);
      max_abs_diff          = std::max(max_abs_diff, std::abs(values_add_host(i) - expected));
      max_abs_val           = std::max(max_abs_val, std::abs(expected));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "integrate_gradients<true> vs values_before + integrate_gradients<false>: dim = "
            << dim << ", nq = " << nq << std::endl;
  std::cout << "  max |expected|      = " << max_abs_val << std::endl;
  std::cout << "  max |add - expected| = " << max_abs_diff << std::endl;
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

  all_passed &= run_integrate_add_test<3, 4>();
  all_passed &= run_integrate_add_test<2, 5>();

  std::cout << (all_passed ? "ALL PASS" : "SOME FAILED") << std::endl;

  return all_passed ? 0 : 1;
}
