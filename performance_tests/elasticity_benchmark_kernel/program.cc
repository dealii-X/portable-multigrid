// The linear elasticity kernel of the dealii-X benchmarks
// (kernels/elasticity_benchmark_kernel.h) on its own, as the baseline for the
// elasticity operators of this repository: element-local vectors, no
// gather/scatter, 10 geometry values per quadrature point. The data are
// those of the benchmark driver (src/kokkos/kokkosLinearElasticityBenchmark.cc),
// so for nq = degree + 2 the check value is the one the benchmark prints.
//
//   ./program degree [n_q_points_1d (degree + 1 or degree + 2, default
//             degree + 2)] [n_elements (default 2^16)] [n_repetitions
//             (default 10)] [n_teams (default: one per batch)]
//
// Timings after one warm-up run; min as in the benchmark, and the median.
// DoFs are element DoFs, 3 (degree + 1)^3 per element.

#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kernels/elasticity_benchmark_kernel.h"

using Number = double;



template <unsigned int nm, unsigned int nq>
int
run(const std::size_t nelmt, const int n_repetitions, const unsigned int requested_teams)
{
  using Kernel                   = Benchmark::LinearElasticityKernel<Number, nm, nq>;
  constexpr unsigned int ncomp   = Kernel::ncomp;
  constexpr unsigned int n_nodes = nm * nm * nm;
  constexpr unsigned int nquad   = nq * nq * nq;

  const Number lambda = 1.;
  const Number mu     = 1.;

  std::vector<Number> basis(nm * nq), dbasis(nq * nq);
  for (unsigned int p = 0; p < nq; ++p)
    for (unsigned int i = 0; i < nm; ++i)
      basis[i * nq + p] = std::sin(i * nq + p);
  for (unsigned int i = 0; i < nq; ++i)
    for (unsigned int p = 0; p < nq; ++p)
      dbasis[i * nq + p] = std::cos(i * nq + p);

  const unsigned int nelmtPerBatch = Kernel::default_elements_per_batch();
  const Kernel       kernel(nelmt, nelmtPerBatch, basis, dbasis, {lambda, mu});
  const std::size_t  nelmt_padded = kernel.n_elements_padded();
  const std::size_t  nbatches     = nelmt_padded / nelmtPerBatch;
  const unsigned int n_teams      = requested_teams > 0 ? requested_teams : nbatches;

  // the benchmark's data, unpadded on the host, the padding zero
  typename Kernel::View d_G("G", nelmt_padded * 10 * nquad);
  typename Kernel::View d_in("in", nelmt_padded * ncomp * n_nodes);
  typename Kernel::View d_out("out", nelmt_padded * ncomp * n_nodes);
  {
    auto G  = Kokkos::create_mirror_view(d_G);
    auto in = Kokkos::create_mirror_view(d_in);
    for (std::size_t i = 0; i < nelmt * 10 * nquad; ++i)
      G(i) = std::cos(i);
    for (unsigned int c = 0; c < ncomp; ++c)
      for (std::size_t i = 0; i < nelmt * n_nodes; ++i)
        in(c * nelmt_padded * n_nodes + i) = std::sin(c * nelmt * n_nodes + i);
    Kokkos::deep_copy(d_G, G);
    Kokkos::deep_copy(d_in, in);
  }

  kernel.apply(d_G, d_in, d_out, n_teams);
  Kokkos::fence();

  std::vector<double> times;
  for (int r = 0; r < n_repetitions; ++r)
    {
      Kokkos::Timer timer;
      kernel.apply(d_G, d_in, d_out, n_teams);
      Kokkos::fence();
      times.push_back(timer.seconds());
    }
  std::sort(times.begin(), times.end());
  const double t_min    = times.front();
  const double t_median = times[times.size() / 2];

  // sum of squares of the valid entries, as the benchmark's check
  Number sum = 0.;
  Kokkos::parallel_reduce(
    nelmt * ncomp * n_nodes,
    KOKKOS_LAMBDA(const std::size_t i, Number &value) {
      const std::size_t c   = i / (nelmt * n_nodes);
      const std::size_t rem = i % (nelmt * n_nodes);
      const Number      v   = d_out(c * nelmt_padded * n_nodes + rem);
      value += v * v;
    },
    sum);

  const double n_dofs  = static_cast<double>(ncomp) * n_nodes * nelmt;
  const double n_quads = static_cast<double>(nquad) * nelmt;
  const double bytes   = (2. * n_dofs + 10. * n_quads) * sizeof(Number);

  std::cout << "degree=" << nm - 1 << " n_q_points_1d=" << nq << " elements=" << nelmt
            << " elements/batch=" << nelmtPerBatch << " teams=" << n_teams
            << " threads/team=" << kernel.team_size()
            << " execution space=" << Kokkos::DefaultExecutionSpace::name() << std::endl
            << "  element dofs " << static_cast<std::size_t>(n_dofs) << ", time min "
            << std::setprecision(4) << 1e3 * t_min << " ms, median " << 1e3 * t_median << " ms"
            << std::endl
            << "  " << 1e-9 * n_dofs / t_min << " GDoFs/s, " << 1e-9 * bytes / t_min
            << " GB/s (benchmark's estimate, min time)" << std::endl
            << "  check " << std::scientific << std::setprecision(8) << std::sqrt(sum)
            << std::defaultfloat << std::endl
            << std::endl;
  return 0;
}



template <unsigned int nm>
int
dispatch_points(const unsigned int n_q_points_1d,
                const std::size_t  nelmt,
                const int          n_repetitions,
                const unsigned int n_teams)
{
  if (n_q_points_1d == nm)
    return run<nm, nm>(nelmt, n_repetitions, n_teams);
  else if (n_q_points_1d == nm + 1)
    return run<nm, nm + 1>(nelmt, n_repetitions, n_teams);
  std::cerr << "n_q_points_1d must be degree + 1 or degree + 2" << std::endl;
  return 1;
}



int
main(int argc, char *argv[])
{
  Kokkos::ScopeGuard kokkos(argc, argv);

  if (argc < 2)
    {
      std::cerr << "usage: " << argv[0]
                << " degree [n_q_points_1d (default degree + 2)] [n_elements (default 2^16)]"
                   " [n_repetitions (default 10)] [n_teams (default: one per batch)]"
                << std::endl;
      return 1;
    }

  const int          degree        = std::atoi(argv[1]);
  const unsigned int n_q_points_1d = (argc > 2) ? std::atoi(argv[2]) : degree + 2;
  const std::size_t  nelmt         = (argc > 3) ? std::atoll(argv[3]) : (1ull << 16);
  const int          n_repetitions = (argc > 4) ? std::atoi(argv[4]) : 10;
  const unsigned int n_teams       = (argc > 5) ? std::atoi(argv[5]) : 0;

  // nm = degree + 1 nodes per direction
  switch (degree)
    {
      case 1:
        return dispatch_points<2>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      case 2:
        return dispatch_points<3>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      case 3:
        return dispatch_points<4>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      case 4:
        return dispatch_points<5>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      case 5:
        return dispatch_points<6>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      case 6:
        return dispatch_points<7>(n_q_points_1d, nelmt, n_repetitions, n_teams);
      default:
        std::cerr << "degree must be in [1, 6]" << std::endl;
        return 1;
    }
}
