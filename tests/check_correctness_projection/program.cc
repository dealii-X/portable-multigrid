// Correctness check for Portable::internal::apply_block_group_mean_projection
// (include/kernels/portable_block_group_mean_projection.h) -- the block
// Pi (homogeneous primal-constraint projector) used by
// SubdomainBDDCOperatorWrapper::project_block().
//
// Unlike the leaf matrix-free kernel and the block-CG solver, this
// operation is purely index-based (group offsets/member dofs/weights),
// with no mesh or matrix-free machinery involved at all -- so this test
// uses synthetic constraint groups (sizes mimicking real vertex/edge/face
// groups) and an independent plain-C++ host reference implementation,
// rather than reconstructing SubdomainBDDCOperatorWrapper's full
// subdomain-triangulation setup.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/portable_block_group_mean_projection.h"

using namespace dealii;
using Number = double;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  // Synthetic primal-constraint groups: sizes mimicking a mix of vertex (1
  // dof), edge, and face constraint entities, deliberately uneven.
  const std::vector<unsigned int> group_sizes = {1, 3, 2, 4, 1, 5, 2};
  const unsigned int              n_groups    = group_sizes.size();

  std::vector<unsigned int> offsets_host(n_groups + 1, 0);
  for (unsigned int g = 0; g < n_groups; ++g)
    offsets_host[g + 1] = offsets_host[g] + group_sizes[g];
  const unsigned int n_members = offsets_host.back();

  // Distinct member dofs [0, n_members), plus a handful of untouched free
  // dofs at the end -- mirrors real subdomains, where not every dof
  // belongs to a primal constraint.
  const unsigned int n_free_dofs = 6;
  const unsigned int dof_stride  = n_members + n_free_dofs;

  std::vector<unsigned int> member_dofs_host(n_members);
  std::iota(member_dofs_host.begin(), member_dofs_host.end(), 0);

  std::vector<Number> weights_host(n_groups);
  for (unsigned int g = 0; g < n_groups; ++g)
    weights_host[g] = Number(1) / Number(group_sizes[g]);

  const unsigned int n_rhs = 6; // K

  // --- build device views ---
  Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> offsets("offsets",
                                                                          n_groups + 1);
  Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> member_dofs("member_dofs",
                                                                               n_members);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> weights("weights", n_groups);

  Kokkos::deep_copy(offsets,
                    Kokkos::View<const unsigned int *, Kokkos::HostSpace>(offsets_host.data(),
                                                                          n_groups + 1));
  Kokkos::deep_copy(member_dofs,
                    Kokkos::View<const unsigned int *, Kokkos::HostSpace>(member_dofs_host.data(),
                                                                          n_members));
  Kokkos::deep_copy(weights,
                    Kokkos::View<const Number *, Kokkos::HostSpace>(weights_host.data(),
                                                                    n_groups));

  // --- random input, k-major block layout (block k occupies
  //     [k*dof_stride, (k+1)*dof_stride)) ---
  std::mt19937                          rng(7);
  std::uniform_real_distribution<Number> dist(-1., 1.);

  std::vector<Number> block_host(static_cast<std::size_t>(n_rhs) * dof_stride);
  for (auto &v : block_host)
    v = dist(rng);

  // --- independent host reference: same math, plain loops ---
  std::vector<Number> reference = block_host;
  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      Number *block = reference.data() + static_cast<std::size_t>(k) * dof_stride;
      for (unsigned int g = 0; g < n_groups; ++g)
        {
          const unsigned int start = offsets_host[g];
          const unsigned int end   = offsets_host[g + 1];
          if (end == start)
            continue;

          Number average = 0;
          for (unsigned int i = start; i < end; ++i)
            average += block[member_dofs_host[i]];
          average *= weights_host[g];

          for (unsigned int i = start; i < end; ++i)
            block[member_dofs_host[i]] -= average;
        }
    }

  // --- device: block kernel under test ---
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> block_vector(
    "block_vector", static_cast<std::size_t>(n_rhs) * dof_stride);
  Kokkos::deep_copy(block_vector,
                    Kokkos::View<const Number *, Kokkos::HostSpace>(block_host.data(),
                                                                    block_host.size()));

  Portable::internal::apply_block_group_mean_projection(
    block_vector.data(), offsets, member_dofs, weights, n_groups, n_rhs, dof_stride);

  auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), block_vector);

  // --- compare ---
  Number max_abs_diff = 0.;
  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      Number column_max_diff = 0.;
      for (unsigned int i = 0; i < dof_stride; ++i)
        {
          const std::size_t idx = static_cast<std::size_t>(k) * dof_stride + i;
          column_max_diff = std::max(column_max_diff, std::abs(reference[idx] - result_host(idx)));
        }
      std::cout << "RHS " << k << ": max |ref - block| = " << column_max_diff << std::endl;
      max_abs_diff = std::max(max_abs_diff, column_max_diff);
    }

  // Also check every group's mean is now exactly zero in every block
  // (the actual mathematical property Pi is supposed to establish), not
  // just agreement with the reference implementation.
  Number max_group_mean = 0.;
  for (unsigned int k = 0; k < n_rhs; ++k)
    for (unsigned int g = 0; g < n_groups; ++g)
      {
        const unsigned int start = offsets_host[g];
        const unsigned int end   = offsets_host[g + 1];
        if (end == start)
          continue;
        Number sum = 0;
        for (unsigned int i = start; i < end; ++i)
          sum += result_host(static_cast<std::size_t>(k) * dof_stride + member_dofs_host[i]);
        max_group_mean = std::max(max_group_mean, std::abs(sum / group_sizes[g]));
      }

  std::cout << std::endl
            << "n_groups = " << n_groups << ", n_members = " << n_members
            << ", dof_stride = " << dof_stride << ", n_rhs = " << n_rhs << std::endl
            << "Overall max |ref - block| = " << max_abs_diff << std::endl
            << "Max |group mean after projection| = " << max_group_mean << std::endl;

  if (max_abs_diff > 1e-12 || max_group_mean > 1e-12)
    {
      std::cout << "FAILED" << std::endl;
      return 1;
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}
