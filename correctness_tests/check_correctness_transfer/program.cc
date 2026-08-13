// Correctness check for BK1Block::Parallel::KokkosProlongationBatchedBlockKernel
// / KokkosRestrictionBatchedBlockKernel (include/kernels/bk1_kokkos_kernels_block.h)
// against K sequential calls of the originals in
// include/kernels/bk1_kokkos_kernels.h (the only two kernels from that file
// actually used in production -- see portable_geometric_transfer.h /
// portable_polynomial_transfer.h).
//
// Like the leaf Laplace kernel check, this only tests whether the (cell, k)
// index-splitting is correct -- not whether the interpolation matrix/weights
// are a real prolongation operator -- so d_shape_values/weights/dof maps are
// synthetic (random, with some invalid coarse dof indices to exercise that
// branch) rather than reconstructed from an actual GeometricTransfer/
// PolynomialTransfer setup, which would require the full transfer-scheme
// machinery those classes build.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include "kernels/bk1_kokkos_kernels.h"
#include "kernels/bk1_kokkos_kernels_block.h"

using namespace dealii;
using Number = double;

constexpr int dim       = 3;
constexpr int nm_coarse = 3; // fe_degree = 2
constexpr int nm_fine   = 5; // 2 * fe_degree + 1

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int n_cells       = 24;
  const unsigned int n_rhs         = 7; // K
  const unsigned int n_coarse_dofs = 60;
  const unsigned int n_fine_dofs   = 260;

  constexpr unsigned int nm_coarse_total = Utilities::pow(nm_coarse, dim);
  constexpr unsigned int nm_fine_total   = Utilities::pow(nm_fine, dim);

  std::mt19937                          rng(99);
  std::uniform_real_distribution<Number> dist(-1., 1.);
  std::uniform_int_distribution<unsigned int> coarse_idx_dist(0, n_coarse_dofs - 1);
  std::uniform_int_distribution<unsigned int> fine_idx_dist(0, n_fine_dofs - 1);
  std::uniform_real_distribution<double>       invalid_chance(0., 1.);

  std::vector<unsigned int> dof_indices_coarse_host(nm_coarse_total * n_cells);
  for (auto &v : dof_indices_coarse_host)
    v = (invalid_chance(rng) < 0.05) ? numbers::invalid_unsigned_int : coarse_idx_dist(rng);

  std::vector<unsigned int> dof_indices_fine_host(nm_fine_total * n_cells);
  for (auto &v : dof_indices_fine_host)
    v = fine_idx_dist(rng);

  std::vector<Number> weights_host(nm_fine_total * n_cells);
  for (auto &v : weights_host)
    v = dist(rng);

  std::vector<Number> shape_values_host(nm_coarse * nm_fine);
  for (auto &v : shape_values_host)
    v = dist(rng);

  Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_coarse(
    "dof_indices_coarse", nm_coarse_total, n_cells);
  Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_fine(
    "dof_indices_fine", nm_fine_total, n_cells);
  Kokkos::View<Number **, MemorySpace::Default::kokkos_space> weights("weights",
                                                                      nm_fine_total,
                                                                      n_cells);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values("shape_values",
                                                                          nm_coarse * nm_fine);

  Kokkos::deep_copy(dof_indices_coarse,
                    Kokkos::View<const unsigned int **, Kokkos::LayoutRight, Kokkos::HostSpace>(
                      dof_indices_coarse_host.data(), nm_coarse_total, n_cells));
  Kokkos::deep_copy(dof_indices_fine,
                    Kokkos::View<const unsigned int **, Kokkos::LayoutRight, Kokkos::HostSpace>(
                      dof_indices_fine_host.data(), nm_fine_total, n_cells));
  Kokkos::deep_copy(weights,
                    Kokkos::View<const Number **, Kokkos::LayoutRight, Kokkos::HostSpace>(
                      weights_host.data(), nm_fine_total, n_cells));
  Kokkos::deep_copy(shape_values,
                    Kokkos::View<const Number *, Kokkos::HostSpace>(shape_values_host.data(),
                                                                    nm_coarse * nm_fine));

  constexpr bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
  unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

  auto random_vector = [&](const unsigned int size) {
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> v("v", size);
    std::vector<Number> v_host(size);
    for (auto &x : v_host)
      x = dist(rng);
    Kokkos::deep_copy(v, Kokkos::View<const Number *, Kokkos::HostSpace>(v_host.data(), size));
    return v;
  };

  Number max_abs_diff = 0.;

  // --- prolongation: coarse -> fine ---
  {
    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> x(n_rhs), dst_ref(n_rhs);
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> x_block("x_block",
                                                                         n_rhs * n_coarse_dofs);
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> dst_block("dst_block",
                                                                           n_rhs * n_fine_dofs);
    Kokkos::deep_copy(dst_block, 0.);

    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        x[k]       = random_vector(n_coarse_dofs);
        dst_ref[k] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>("dst_ref", n_fine_dofs);
        Kokkos::deep_copy(dst_ref[k], 0.);

        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x[k]);
        Kokkos::deep_copy(Kokkos::subview(x_block,
                                          Kokkos::make_pair(k * n_coarse_dofs,
                                                            (k + 1) * n_coarse_dofs)),
                          x_host);

        BK1::Parallel::KokkosProlongationBatchedKernel<dim, nm_coarse, nm_fine, Number>(
          shape_values,
          x[k],
          dst_ref[k],
          dof_indices_coarse,
          dof_indices_fine,
          weights,
          n_cells,
          numBlocks,
          threadsPerBlock);
      }
    Kokkos::fence();

    BK1Block::Parallel::KokkosProlongationBatchedBlockKernel<dim, nm_coarse, nm_fine, Number>(
      shape_values,
      x_block,
      dst_block,
      dof_indices_coarse,
      dof_indices_fine,
      weights,
      n_cells,
      n_rhs,
      n_coarse_dofs,
      n_fine_dofs,
      numBlocks,
      threadsPerBlock);
    Kokkos::fence();

    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        auto ref_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_ref[k]);
        auto block_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          Kokkos::subview(dst_block,
                          Kokkos::make_pair(k * n_fine_dofs, (k + 1) * n_fine_dofs)));

        Number column_max_diff = 0.;
        for (unsigned int i = 0; i < n_fine_dofs; ++i)
          column_max_diff = std::max(column_max_diff, std::abs(ref_host(i) - block_host(i)));

        std::cout << "Prolongation RHS " << k << ": max |ref - block| = " << column_max_diff
                  << std::endl;
        max_abs_diff = std::max(max_abs_diff, column_max_diff);
      }
  }

  // --- restriction: fine -> coarse ---
  {
    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> x(n_rhs), dst_ref(n_rhs);
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> x_block("x_block",
                                                                         n_rhs * n_fine_dofs);
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> dst_block("dst_block",
                                                                           n_rhs * n_coarse_dofs);
    Kokkos::deep_copy(dst_block, 0.);

    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        x[k]       = random_vector(n_fine_dofs);
        dst_ref[k] =
          Kokkos::View<Number *, MemorySpace::Default::kokkos_space>("dst_ref", n_coarse_dofs);
        Kokkos::deep_copy(dst_ref[k], 0.);

        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x[k]);
        Kokkos::deep_copy(Kokkos::subview(x_block,
                                          Kokkos::make_pair(k * n_fine_dofs, (k + 1) * n_fine_dofs)),
                          x_host);

        BK1::Parallel::KokkosRestrictionBatchedKernel<dim, nm_coarse, nm_fine, Number>(
          shape_values,
          x[k],
          dst_ref[k],
          dof_indices_coarse,
          dof_indices_fine,
          weights,
          n_cells,
          numBlocks,
          threadsPerBlock);
      }
    Kokkos::fence();

    BK1Block::Parallel::KokkosRestrictionBatchedBlockKernel<dim, nm_coarse, nm_fine, Number>(
      shape_values,
      x_block,
      dst_block,
      dof_indices_coarse,
      dof_indices_fine,
      weights,
      n_cells,
      n_rhs,
      n_coarse_dofs,
      n_fine_dofs,
      numBlocks,
      threadsPerBlock);
    Kokkos::fence();

    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        auto ref_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_ref[k]);
        auto block_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          Kokkos::subview(dst_block,
                          Kokkos::make_pair(k * n_coarse_dofs, (k + 1) * n_coarse_dofs)));

        Number column_max_diff = 0.;
        for (unsigned int i = 0; i < n_coarse_dofs; ++i)
          column_max_diff = std::max(column_max_diff, std::abs(ref_host(i) - block_host(i)));

        std::cout << "Restriction RHS " << k << ": max |ref - block| = " << column_max_diff
                  << std::endl;
        max_abs_diff = std::max(max_abs_diff, column_max_diff);
      }
  }

  std::cout << std::endl << "Overall max |ref - block| = " << max_abs_diff << std::endl;

  if (max_abs_diff > 1e-9)
    {
      std::cout << "FAILED" << std::endl;
      return 1;
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}
