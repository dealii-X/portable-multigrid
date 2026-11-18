// Custom::Parallel::FEEvaluationImplTransformToCollocation
// (matrix_free/portable_evaluation_kernels.h) on the host: evaluate_gradients()
// against evaluate_gradients_and_multiply_symmetric_tensor() with G = I, the
// add variants of evaluate/integrate against their two-step forms, and
// evaluate()/integrate() against values + gradients separately.

#include "../tests.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include "matrix_free/portable_evaluation_kernels.h"

using namespace dealii;

using Number = double;

template <int dim, int nq>
bool
run_evaluate_gradients_test()
{
  constexpr int nq_total                   = Utilities::pow(nq, dim);
  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients(
    "co_shape_gradients", nq * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values("values", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> d_G_identity(
    "d_G_identity", symmetric_tensor_dimension * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> grad_a("grad_a", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> grad_b("grad_b", dim * nq_total);

  {
    std::mt19937                           gen(11);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto values_host = Kokkos::create_mirror_view(values);
    for (int i = 0; i < nq_total; ++i)
      values_host(i) = dist(gen);
    Kokkos::deep_copy(values, values_host);

    auto d_G_identity_host = Kokkos::create_mirror_view(d_G_identity);
    int  component_index   = 0;
    for (int d1 = 0; d1 < dim; ++d1)
      for (int d2 = d1; d2 < dim; ++d2)
        {
          for (int q = 0; q < nq_total; ++q)
            d_G_identity_host(component_index * nq_total + q) = (d1 == d2) ? 1. : 0.;
          ++component_index;
        }
    Kokkos::deep_copy(d_G_identity, d_G_identity_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int thread_id = team_member.team_rank();
      const int block_size = team_member.team_size();
      const int batch_index  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nq, nq, Number> fe_eval(
        team_member, nullptr, co_shape_gradients.data(), 1, 1, batch_index, thread_id, block_size);

      fe_eval.evaluate_gradients(values.data(), grad_a.data());
      fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(
        d_G_identity, Custom::Parallel::CellRangeIdView(), values.data(), grad_b.data());
    });
  Kokkos::fence();

  auto grad_a_host = Kokkos::create_mirror_view(grad_a);
  auto grad_b_host = Kokkos::create_mirror_view(grad_b);
  Kokkos::deep_copy(grad_a_host, grad_a);
  Kokkos::deep_copy(grad_b_host, grad_b);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (int i = 0; i < dim * nq_total; ++i)
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(grad_a_host(i) - grad_b_host(i)));
      max_abs_val  = std::max(max_abs_val, std::abs(grad_a_host(i)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  deallog << "evaluate_gradients() vs evaluate_gradients_and_multiply_symmetric_tensor()+identity: "
            << "dim = " << dim << ", nq = " << nq << std::endl;
  if (pass)
    deallog << "OK" << std::endl;
  else
    deallog << "max diff = " << max_abs_diff << ", max value = " << max_abs_val << std::endl;

  return pass;
}

// Isolates the `add` template parameter on evaluate_gradients() and
// evaluate_gradients_and_multiply_symmetric_tensor(): the <true> result
// should equal out_before + the <false> result, for both methods, checked
// together since they share the same reference-space contraction.
template <int dim, int nq>
bool
run_evaluate_gradients_add_test()
{
  constexpr int nq_total                   = Utilities::pow(nq, dim);
  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients(
    "co_shape_gradients", nq * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values("values", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> d_G_identity(
    "d_G_identity", symmetric_tensor_dimension * nq_total);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_preexisting(
    "out_preexisting", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_noadd("out_noadd", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_add("out_add", dim * nq_total);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_mult_preexisting(
    "out_mult_preexisting", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_mult_noadd(
    "out_mult_noadd", dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_mult_add(
    "out_mult_add", dim * nq_total);

  {
    std::mt19937                           gen(13);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto values_host = Kokkos::create_mirror_view(values);
    for (int i = 0; i < nq_total; ++i)
      values_host(i) = dist(gen);
    Kokkos::deep_copy(values, values_host);

    auto d_G_identity_host = Kokkos::create_mirror_view(d_G_identity);
    int  component_index   = 0;
    for (int d1 = 0; d1 < dim; ++d1)
      for (int d2 = d1; d2 < dim; ++d2)
        {
          for (int q = 0; q < nq_total; ++q)
            d_G_identity_host(component_index * nq_total + q) = (d1 == d2) ? 1. : 0.;
          ++component_index;
        }
    Kokkos::deep_copy(d_G_identity, d_G_identity_host);

    auto out_preexisting_host      = Kokkos::create_mirror_view(out_preexisting);
    auto out_add_host              = Kokkos::create_mirror_view(out_add);
    auto out_mult_preexisting_host = Kokkos::create_mirror_view(out_mult_preexisting);
    auto out_mult_add_host         = Kokkos::create_mirror_view(out_mult_add);
    for (int i = 0; i < dim * nq_total; ++i)
      {
        const Number v                  = dist(gen);
        out_add_host(i)                 = out_preexisting_host(i) = v;
        out_mult_add_host(i) = out_mult_preexisting_host(i) = v;
      }
    Kokkos::deep_copy(out_preexisting, out_preexisting_host);
    Kokkos::deep_copy(out_add, out_add_host);
    Kokkos::deep_copy(out_mult_preexisting, out_mult_preexisting_host);
    Kokkos::deep_copy(out_mult_add, out_mult_add_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int thread_id = team_member.team_rank();
      const int block_size = team_member.team_size();
      const int batch_index  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nq, nq, Number> fe_eval(
        team_member, nullptr, co_shape_gradients.data(), 1, 1, batch_index, thread_id, block_size);

      fe_eval.template evaluate_gradients<false>(values.data(), out_noadd.data());
      fe_eval.template evaluate_gradients<true>(values.data(), out_add.data());

      fe_eval.template evaluate_gradients_and_multiply_symmetric_tensor<false>(
        d_G_identity, Custom::Parallel::CellRangeIdView(), values.data(), out_mult_noadd.data());
      fe_eval.template evaluate_gradients_and_multiply_symmetric_tensor<true>(
        d_G_identity, Custom::Parallel::CellRangeIdView(), values.data(), out_mult_add.data());
    });
  Kokkos::fence();

  auto out_preexisting_host      = Kokkos::create_mirror_view(out_preexisting);
  auto out_add_host              = Kokkos::create_mirror_view(out_add);
  auto out_noadd_host             = Kokkos::create_mirror_view(out_noadd);
  auto out_mult_preexisting_host = Kokkos::create_mirror_view(out_mult_preexisting);
  auto out_mult_add_host         = Kokkos::create_mirror_view(out_mult_add);
  auto out_mult_noadd_host        = Kokkos::create_mirror_view(out_mult_noadd);
  Kokkos::deep_copy(out_preexisting_host, out_preexisting);
  Kokkos::deep_copy(out_add_host, out_add);
  Kokkos::deep_copy(out_noadd_host, out_noadd);
  Kokkos::deep_copy(out_mult_preexisting_host, out_mult_preexisting);
  Kokkos::deep_copy(out_mult_add_host, out_mult_add);
  Kokkos::deep_copy(out_mult_noadd_host, out_mult_noadd);

  double max_abs_diff = 0., max_abs_val = 0.;
  for (int i = 0; i < dim * nq_total; ++i)
    {
      const double expected = out_preexisting_host(i) + out_noadd_host(i);
      max_abs_diff = std::max(max_abs_diff, std::abs(out_add_host(i) - expected));
      max_abs_val  = std::max(max_abs_val, std::abs(expected));

      const double expected_mult = out_mult_preexisting_host(i) + out_mult_noadd_host(i);
      max_abs_diff = std::max(max_abs_diff, std::abs(out_mult_add_host(i) - expected_mult));
      max_abs_val  = std::max(max_abs_val, std::abs(expected_mult));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  deallog << "evaluate_gradients<true>/evaluate_gradients_and_multiply_symmetric_tensor<true> "
            << "vs out_before + <false>: dim = " << dim << ", nq = " << nq << std::endl;
  if (pass)
    deallog << "OK" << std::endl;
  else
    deallog << "max diff = " << max_abs_diff << ", max value = " << max_abs_val << std::endl;

  return pass;
}

// Isolated check for FEEvaluationImplTransformToCollocation::
// integrate_gradients() (also lightly hand-edited recently): isolates the
// `add` template parameter -- integrate_gradients<true>(..., values) should
// equal values_before + integrate_gradients<false>(..., values) -- same
// technique as the (now removed/stale-binary) copy of this test in
// check_correctness_batched_fe_evaluation/program.cc, rebuilt fresh here so
// it isn't shadowed by that target's own unrelated build failure.
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
      const int thread_id = team_member.team_rank();
      const int block_size = team_member.team_size();
      const int batch_index  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nq, nq, Number> fe_eval(
        team_member, nullptr, co_shape_gradients.data(), 1, 1, batch_index, thread_id, block_size);

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

  deallog << "integrate_gradients<true> vs values_before + integrate_gradients<false>: dim = "
            << dim << ", nq = " << nq << std::endl;
  if (pass)
    deallog << "OK" << std::endl;
  else
    deallog << "max diff = " << max_abs_diff << ", max value = " << max_abs_val << std::endl;

  return pass;
}

// Cross-checks the new unified evaluate() (mirroring real deal.II's
// FEEvaluationImplTransformToCollocation::evaluate(), using
// EvaluatorTensorProduct's in_place values<>/co_gradients<> machinery)
// against the already-validated hand-written primitives
// evaluate_values()/evaluate_gradients(). fe_degree may differ from
// nq - 1 (over-integration); nq must be >= fe_degree + 1 so a single
// buffer sized to the quadrature-point count can hold every intermediate
// stage of both paths.
template <int dim, int fe_degree, int nq>
bool
run_evaluate_roundtrip_test(bool use_gradients_flag)
{
  static_assert(nq >= fe_degree + 1, "test buffers assume nq >= fe_degree + 1");

  constexpr int n_dofs_1d    = fe_degree + 1;
  constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);
  constexpr int nq_total     = Utilities::pow(nq, dim);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values("shape_values",
                                                                          n_dofs_1d * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients(
    "co_shape_gradients", nq * nq);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_a("values_a", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> gradients_a("gradients_a",
                                                                          dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> scratch_a("scratch_a", 2 * nq_total);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_b("values_b", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> gradients_b("gradients_b",
                                                                          dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> scratch_b("scratch_b", nq_total);

  {
    std::mt19937                           gen(23);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto shape_values_host = Kokkos::create_mirror_view(shape_values);
    for (int i = 0; i < n_dofs_1d * nq; ++i)
      shape_values_host(i) = dist(gen);
    Kokkos::deep_copy(shape_values, shape_values_host);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto values_a_host = Kokkos::create_mirror_view(values_a);
    auto values_b_host = Kokkos::create_mirror_view(values_b);
    for (int i = 0; i < n_dofs_total; ++i)
      values_a_host(i) = values_b_host(i) = dist(gen);
    Kokkos::deep_copy(values_a, values_a_host);
    Kokkos::deep_copy(values_b, values_b_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int thread_id = team_member.team_rank();
      const int block_size = team_member.team_size();
      const int batch_index  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, fe_degree, nq, Number>
        fe_eval(team_member,
                shape_values.data(),
                co_shape_gradients.data(),
                1,
                1,
                batch_index,
                thread_id,
                block_size);

      // Path A: existing, already-validated hand-written primitives.
      fe_eval.evaluate_values(values_a.data(), values_a.data(), scratch_a.data());
      if (use_gradients_flag)
        fe_eval.evaluate_gradients(values_a.data(), gradients_a.data());

      // Path B: new unified evaluate(), mirroring real deal.II.
      fe_eval.evaluate(values_b.data(),
                       gradients_b.data(),
                       scratch_b.data(),
                       use_gradients_flag ? (EvaluationFlags::values | EvaluationFlags::gradients) :
                                            EvaluationFlags::values);
    });
  Kokkos::fence();

  auto values_a_host    = Kokkos::create_mirror_view(values_a);
  auto values_b_host    = Kokkos::create_mirror_view(values_b);
  auto gradients_a_host = Kokkos::create_mirror_view(gradients_a);
  auto gradients_b_host = Kokkos::create_mirror_view(gradients_b);
  Kokkos::deep_copy(values_a_host, values_a);
  Kokkos::deep_copy(values_b_host, values_b);
  Kokkos::deep_copy(gradients_a_host, gradients_a);
  Kokkos::deep_copy(gradients_b_host, gradients_b);

  double max_abs_diff = 0., max_abs_val = 0.;
  for (int i = 0; i < nq_total; ++i)
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(values_a_host(i) - values_b_host(i)));
      max_abs_val  = std::max(max_abs_val, std::abs(values_a_host(i)));
    }
  if (use_gradients_flag)
    for (int i = 0; i < dim * nq_total; ++i)
      {
        max_abs_diff =
          std::max(max_abs_diff, std::abs(gradients_a_host(i) - gradients_b_host(i)));
        max_abs_val = std::max(max_abs_val, std::abs(gradients_a_host(i)));
      }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  deallog << "evaluate() vs evaluate_values()+evaluate_gradients(): dim = " << dim
            << ", fe_degree = " << fe_degree << ", nq = " << nq
            << ", gradients_flag = " << (use_gradients_flag ? "true" : "false") << std::endl;
  if (pass)
    deallog << "OK" << std::endl;
  else
    deallog << "max diff = " << max_abs_diff << ", max value = " << max_abs_val << std::endl;

  return pass;
}

// Cross-checks the new unified integrate() against the already-validated
// hand-written integrate_gradients()/integrate_values() composition, for
// both integration_flag == gradients and integration_flag == gradients |
// values (the latter accumulating a pre-existing quadrature-space
// "values" contribution, e.g. a mass-matrix term, on top of the gradient
// contribution -- matches integrate_gradients<true>()'s add semantics).
template <int dim, int fe_degree, int nq>
bool
run_integrate_roundtrip_test(bool use_values_flag)
{
  static_assert(nq >= fe_degree + 1, "test buffers assume nq >= fe_degree + 1");

  constexpr int n_dofs_1d    = fe_degree + 1;
  constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);
  constexpr int nq_total     = Utilities::pow(nq, dim);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values("shape_values",
                                                                          n_dofs_1d * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients(
    "co_shape_gradients", nq * nq);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> gradients("gradients",
                                                                        dim * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> gradients_b("gradients_b",
                                                                          dim * nq_total);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_a("values_a", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> scratch_a("scratch_a", 2 * nq_total);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> values_b("values_b", nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> scratch_b("scratch_b", nq_total);

  {
    std::mt19937                           gen(29);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto shape_values_host = Kokkos::create_mirror_view(shape_values);
    for (int i = 0; i < n_dofs_1d * nq; ++i)
      shape_values_host(i) = dist(gen);
    Kokkos::deep_copy(shape_values, shape_values_host);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto gradients_host = Kokkos::create_mirror_view(gradients);
    for (int i = 0; i < dim * nq_total; ++i)
      gradients_host(i) = dist(gen);
    Kokkos::deep_copy(gradients, gradients_host);
    Kokkos::deep_copy(gradients_b, gradients_host);

    auto values_a_host = Kokkos::create_mirror_view(values_a);
    auto values_b_host = Kokkos::create_mirror_view(values_b);
    for (int i = 0; i < nq_total; ++i)
      values_a_host(i) = values_b_host(i) = use_values_flag ? dist(gen) : 0.;
    Kokkos::deep_copy(values_a, values_a_host);
    Kokkos::deep_copy(values_b, values_b_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int thread_id = team_member.team_rank();
      const int block_size = team_member.team_size();
      const int batch_index  = team_member.league_rank();

      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, fe_degree, nq, Number>
        fe_eval(team_member,
                shape_values.data(),
                co_shape_gradients.data(),
                1,
                1,
                batch_index,
                thread_id,
                block_size);

      // Path A: existing, already-validated hand-written primitives.
      if (use_values_flag)
        fe_eval.template integrate_gradients<true>(gradients.data(), values_a.data());
      else
        fe_eval.template integrate_gradients<false>(gradients.data(), values_a.data());
      fe_eval.integrate_values(values_a.data(), values_a.data(), scratch_a.data());

      // Path B: new unified integrate(), mirroring real deal.II.
      fe_eval.integrate(values_b.data(),
                        gradients_b.data(),
                        scratch_b.data(),
                        use_values_flag ? (EvaluationFlags::values | EvaluationFlags::gradients) :
                                          EvaluationFlags::gradients);
    });
  Kokkos::fence();

  auto values_a_host = Kokkos::create_mirror_view(values_a);
  auto values_b_host = Kokkos::create_mirror_view(values_b);
  Kokkos::deep_copy(values_a_host, values_a);
  Kokkos::deep_copy(values_b_host, values_b);

  double max_abs_diff = 0., max_abs_val = 0.;
  for (int i = 0; i < n_dofs_total; ++i)
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(values_a_host(i) - values_b_host(i)));
      max_abs_val  = std::max(max_abs_val, std::abs(values_a_host(i)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  deallog << "integrate() vs integrate_gradients()+integrate_values(): dim = " << dim
            << ", fe_degree = " << fe_degree << ", nq = " << nq
            << ", values_flag = " << (use_values_flag ? "true" : "false") << std::endl;
  if (pass)
    deallog << "OK" << std::endl;
  else
    deallog << "max diff = " << max_abs_diff << ", max value = " << max_abs_val << std::endl;

  return pass;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();

  bool all_passed = true;
  all_passed &= run_evaluate_gradients_test<3, 4>();
  all_passed &= run_evaluate_gradients_test<2, 5>();
  all_passed &= run_evaluate_gradients_test<1, 3>();
  all_passed &= run_evaluate_gradients_test<3, 2>();
  all_passed &= run_evaluate_gradients_test<2, 2>();

  all_passed &= run_evaluate_gradients_add_test<3, 4>();
  all_passed &= run_evaluate_gradients_add_test<2, 5>();
  all_passed &= run_evaluate_gradients_add_test<1, 3>();
  all_passed &= run_evaluate_gradients_add_test<3, 2>();
  all_passed &= run_evaluate_gradients_add_test<2, 2>();

  all_passed &= run_integrate_add_test<3, 4>();
  all_passed &= run_integrate_add_test<2, 5>();
  all_passed &= run_integrate_add_test<1, 3>();
  all_passed &= run_integrate_add_test<3, 2>();
  all_passed &= run_integrate_add_test<2, 2>();

  // dim, fe_degree, nq: square cases (nq == fe_degree + 1) ...
  all_passed &= run_evaluate_roundtrip_test<3, 3, 4>(true);
  all_passed &= run_evaluate_roundtrip_test<3, 3, 4>(false);
  all_passed &= run_evaluate_roundtrip_test<2, 4, 5>(true);
  all_passed &= run_evaluate_roundtrip_test<2, 4, 5>(false);
  all_passed &= run_evaluate_roundtrip_test<1, 2, 3>(true);
  all_passed &= run_evaluate_roundtrip_test<1, 2, 3>(false);
  all_passed &= run_evaluate_roundtrip_test<2, 1, 2>(true);
  all_passed &= run_evaluate_roundtrip_test<2, 1, 2>(false);
  // ... and one over-integrated case (nq > fe_degree + 1).
  all_passed &= run_evaluate_roundtrip_test<3, 2, 4>(true);
  all_passed &= run_evaluate_roundtrip_test<3, 2, 4>(false);

  all_passed &= run_integrate_roundtrip_test<3, 3, 4>(true);
  all_passed &= run_integrate_roundtrip_test<3, 3, 4>(false);
  all_passed &= run_integrate_roundtrip_test<2, 4, 5>(true);
  all_passed &= run_integrate_roundtrip_test<2, 4, 5>(false);
  all_passed &= run_integrate_roundtrip_test<1, 2, 3>(true);
  all_passed &= run_integrate_roundtrip_test<1, 2, 3>(false);
  all_passed &= run_integrate_roundtrip_test<2, 1, 2>(true);
  all_passed &= run_integrate_roundtrip_test<2, 1, 2>(false);
  all_passed &= run_integrate_roundtrip_test<3, 2, 4>(true);
  all_passed &= run_integrate_roundtrip_test<3, 2, 4>(false);

  (void)all_passed;
}
