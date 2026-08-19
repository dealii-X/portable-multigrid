// Correctness check for Custom::Parallel::FEEvaluation
// (include/kernels/portable_batched_fe_evaluation.h) -- the batched,
// deal.II-Portable::FEEvaluation-faithful get_value()/get_gradient()/
// submit_value()/submit_gradient() accessor API, meant to be called from a
// per-quadrature-point Functor exactly like step-64's HelmholtzOperatorQuad.
//
// Two independent checks:
//
//  1. run_test(): builds a real Portable::LaplaceOperator for ground truth
//     (vmult(), the already-validated unbatched path) and, on the exact same
//     Portable::MatrixFree instance, drives cell_loop_batched_launch()
//     (include/kernels/portable_local_laplace_operator_batched.h) with a
//     small LocalLaplaceOperatorGeneric Functor that computes the identical
//     stiffness operator purely via
//     read_dof_values()/evaluate(gradients)/get_gradient()/submit_gradient()/
//     integrate(gradients)/distribute_local_to_global() -- i.e. the
//     on-the-fly inv_jacobian/JxW path, not LocalLaplaceOperatorBatched's
//     G_tensor-fused one. Compares the two dst vectors on random input.
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

// Batched analog of step-64's LocalHelmholtzOperator, minus the mass term
// (coef == 0), so it computes exactly the same isotropic Laplace stiffness
// operator as Portable::LaplaceOperator -- built purely from
// Custom::Parallel::FEEvaluation's accessor API instead of
// LocalLaplaceOperatorBatched's G_tensor-fused evaluate_gradients_and_
// multiply_symmetric_tensor().
template <int dim, int fe_degree, int n_q_points_1d, typename Number>
class LocalLaplaceOperatorGeneric
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(const Custom::Parallel::BatchData<dim, Number> *data,
             const Custom::Parallel::DeviceView<Number>     &src,
             Custom::Parallel::DeviceView<Number>           &dst) const
  {
    Custom::Parallel::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);

    data->for_each_quad_point([&](const int point)
                                { fe_eval.submit_gradient(fe_eval.get_gradient(point), point); });

    fe_eval.integrate(EvaluationFlags::gradients);

    fe_eval.distribute_local_to_global(dst);
  }
};

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

  AffineConstraints<Number> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  Portable::LaplaceOperator<dim, fe_degree, Number> laplace_operator(
    dof_handler, constraints, /*overlap_communication_computation=*/false);

  const auto        &matrix_free   = laplace_operator.get_matrix_free();
  const auto        &colored_graph = matrix_free.get_colored_graph();
  const unsigned int n_colors      = colored_graph.size();

  // -- setup_dof_indices_per_color(), following
  //    Portable::LaplaceOperator::setup_dof_indices_per_color() -- can't
  //    reach the operator's own copy (private), so duplicated here, same as
  //    correctness_tests/check_correctness_common_kernels/program.cc does. --
  constexpr int             n_1d         = fe_degree + 1;
  constexpr unsigned int    n_local_dofs = Utilities::pow(n_1d, dim);
  std::vector<unsigned int> lex_numbering(n_local_dofs);
  {
    const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
    lex_numbering = shape_info.lexicographic_numbering;
  }

  std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
    dof_indices_per_color(n_colors);

  {
    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
    std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() == 0)
          continue;

        const auto &mf_data = matrix_free.get_data(color);
        const auto &graph   = colored_graph[color];

        dof_indices_per_color[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices_" + std::to_string(color), Kokkos::WithoutInitializing),
            n_local_dofs,
            mf_data.n_cells);

        auto dof_indices_host = Kokkos::create_mirror_view(dof_indices_per_color[color]);

        for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
          {
            auto triacell = graph[cell_id];

            typename DoFHandler<dim>::cell_iterator cell =
              triacell->as_dof_handler_iterator(dof_handler);

            cell->get_dof_indices(local_dof_indices);
            triacell->get_dof_indices(subdomain_local_dof_indices);

            if (partitioner)
              for (auto &index : local_dof_indices)
                index = partitioner->global_to_local(index);

            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                const auto global_dof          = local_dof_indices[lex_numbering[i]];
                const auto subdomain_local_dof = subdomain_local_dof_indices[lex_numbering[i]];

                if (constraints.is_constrained(subdomain_local_dof))
                  dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                else
                  dof_indices_host(i, cell_id) = global_dof;
              }
          }

        Kokkos::deep_copy(dof_indices_per_color[color], dof_indices_host);
        Kokkos::fence();
      }
  }

  // -- random input, apply both paths, compare --
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src, dst_orig, dst_generic;
  laplace_operator.initialize_dof_vector(src);
  laplace_operator.initialize_dof_vector(dst_orig);
  laplace_operator.initialize_dof_vector(dst_generic);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw(src.locally_owned_elements());
    for (const auto idx : src.locally_owned_elements())
      rw(idx) = dist(gen);
    src.import_elements(rw, VectorOperation::insert);
  }

  laplace_operator.vmult(dst_orig, src);

  {
    dst_generic = 0.;

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks       = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
    if (is_serial)
      {
        numBlocks       = 1u;
        threadsPerBlock = 1u;
      }

    // No BK3-style G_tensor is used by LocalLaplaceOperatorGeneric (it goes
    // through FEEvaluation::get_gradient()/submit_gradient()'s on-the-fly
    // inv_jacobian path instead), so an empty placeholder View is enough --
    // LaplaceOperatorBatchData::G_tensor is simply never read on this path.
    const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> unused_G_tensor;

    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const auto &gpu_data = matrix_free.get_data(color, 0);
        if (gpu_data.n_cells > 0)
          Portable::cell_loop_batched_launch<dim, fe_degree, fe_degree + 1, Number>(
            LocalLaplaceOperatorGeneric<dim, fe_degree, fe_degree + 1, Number>{},
            gpu_data,
            dof_indices_per_color[color],
            unused_G_tensor,
            src,
            dst_generic,
            numBlocks,
            threadsPerBlock);
      }

    dst_generic.compress(VectorOperation::add);
    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst_generic);
  }

  LinearAlgebra::ReadWriteVector<Number> rw_orig(dst_orig.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_generic(dst_generic.locally_owned_elements());
  rw_orig.import_elements(dst_orig, VectorOperation::insert);
  rw_generic.import_elements(dst_generic, VectorOperation::insert);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (const auto idx : dst_orig.locally_owned_elements())
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(rw_orig(idx) - rw_generic(idx)));
      max_abs_val  = std::max(max_abs_val, std::abs(rw_orig(idx)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_dofs = " << dof_handler.n_dofs() << std::endl;
  std::cout << "  max |vmult|                = " << max_abs_val << std::endl;
  std::cout << "  max |vmult - fe_eval_based| = " << max_abs_diff << std::endl;
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
