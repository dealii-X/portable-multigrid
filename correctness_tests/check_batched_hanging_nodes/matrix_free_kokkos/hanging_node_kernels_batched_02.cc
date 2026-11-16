// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Test Portable::internal::batched::resolve_hanging_nodes(), which resolves
// the hanging-node constraints of a batch of cells stored cell after cell in
// shared memory, against Portable::internal::resolve_hanging_nodes() applied
// cell by cell. Every combination of the ConstraintKinds bits (subcell,
// face and, in 3d, edge flags) is one cell with random values; the 1d
// interpolation weights are random as well. The cells are distributed over
// teams such that the last team gets a shorter batch, and then once more
// with a single cell per team. The expected output of the first part was
// recorded with the cell-by-cell kernel before the batched one existed; the
// single-cell part was appended later, its checksums also come from the
// cell-by-cell kernel.

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/portable_hanging_nodes_internal.h>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#include "../tests.h"


using ConstraintKinds = dealii::internal::MatrixFreeFunctions::ConstraintKinds;

using TeamHandle = Kokkos::TeamPolicy<
  MemorySpace::Default::kokkos_space::execution_space>::member_type;

template <typename T>
using DeviceView = Kokkos::View<T, MemorySpace::Default::kokkos_space>;

using ScratchView = Kokkos::View<
  double *,
  MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;



// One team resolves the hanging nodes of the cells [first, first + n) of its
// batch twice: cell by cell with resolve_hanging_nodes(), and all cells at
// once with batched::resolve_hanging_nodes().
template <int dim, int fe_degree, bool transpose>
struct ResolveBatch
{
  static constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

  DeviceView<double *>          weights;
  DeviceView<ConstraintKinds *> masks;
  DeviceView<double *>          src;
  DeviceView<double *>          dst_cell_by_cell;
  DeviceView<double *>          dst_batched;
  int                           n_cells;
  int                           cells_per_team;

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    const int first_cell       = team_member.league_rank() * cells_per_team;
    const int n_cells_in_batch = (n_cells - first_cell < cells_per_team) ?
                                   (n_cells - first_cell) :
                                   cells_per_team;
    const int n_values         = n_cells_in_batch * n_dofs;

    ScratchView values(team_member.team_shmem(), cells_per_team * n_dofs);

    const auto copy = [&](const auto from,
                          auto       to,
                          const int  offset_from,
                          const int  offset_to) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_values),
                           [&](const int i) {
                             to(offset_to + i) = from(offset_from + i);
                           });
      team_member.team_barrier();
    };

    // cell by cell
    copy(src, values, first_cell * n_dofs, 0);
    for (int cell = 0; cell < n_cells_in_batch; ++cell)
      if (masks(first_cell + cell) != ConstraintKinds::unconstrained)
        Portable::internal::resolve_hanging_nodes<dim, fe_degree, transpose>(
          team_member,
          weights,
          masks(first_cell + cell),
          Kokkos::subview(
            values, Kokkos::make_pair(cell * n_dofs, (cell + 1) * n_dofs)));
    copy(values, dst_cell_by_cell, 0, first_cell * n_dofs);

    // batched
    copy(src, values, first_cell * n_dofs, 0);
    Portable::internal::batched::
      resolve_hanging_nodes<dim, fe_degree, transpose>(
        team_member,
        weights,
        [&](const int cell) { return masks(first_cell + cell); },
        n_cells_in_batch,
        values);
    copy(values, dst_batched, 0, first_cell * n_dofs);
  }
};



template <int dim, int fe_degree, bool transpose>
void
test(const int cells_per_team)
{
  constexpr int n_dofs_1d = fe_degree + 1;
  constexpr int n_dofs    = Utilities::pow(n_dofs_1d, dim);

  // all combinations of the subcell, face and (3d) edge bits
  constexpr unsigned int n_bits    = (dim == 2) ? 4 : 9;
  constexpr unsigned int n_cells   = 1u << n_bits;
  const auto             make_mask = [](const unsigned int bits) {
    unsigned int mask = 0;
    if (dim == 2)
      {
        // bits: subcell_x, subcell_y, face_x, face_y
        mask |= (bits & 1) ?
                              static_cast<unsigned int>(ConstraintKinds::subcell_x) :
                              0;
        mask |= (bits & 2) ?
                              static_cast<unsigned int>(ConstraintKinds::subcell_y) :
                              0;
        mask |=
          (bits & 4) ? static_cast<unsigned int>(ConstraintKinds::face_x) : 0;
        mask |=
          (bits & 8) ? static_cast<unsigned int>(ConstraintKinds::face_y) : 0;
      }
    else
      // the nine bits of ConstraintKinds in order
      mask = bits;
    return static_cast<ConstraintKinds>(mask);
  };

  deallog << "dim=" << dim << " degree=" << fe_degree
          << " transpose=" << transpose << ": n_cells=" << n_cells
          << " cells_per_team=" << cells_per_team << std::endl;

  DeviceView<double *>          weights("weights", n_dofs_1d * n_dofs_1d);
  DeviceView<ConstraintKinds *> masks("masks", n_cells);
  DeviceView<double *>          src("src", n_cells * n_dofs);
  DeviceView<double *> dst_cell_by_cell("dst_cell_by_cell", n_cells * n_dofs);
  DeviceView<double *> dst_batched("dst_batched", n_cells * n_dofs);

  {
    auto weights_host = Kokkos::create_mirror_view(weights);
    for (unsigned int i = 0; i < weights_host.size(); ++i)
      weights_host(i) =
        -1. + 2. * static_cast<double>(Testing::rand()) / RAND_MAX;
    Kokkos::deep_copy(weights, weights_host);

    auto masks_host = Kokkos::create_mirror_view(masks);
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      masks_host(cell) = make_mask(cell);
    Kokkos::deep_copy(masks, masks_host);

    auto src_host = Kokkos::create_mirror_view(src);
    for (unsigned int i = 0; i < src_host.size(); ++i)
      src_host(i) = -1. + 2. * static_cast<double>(Testing::rand()) / RAND_MAX;
    Kokkos::deep_copy(src, src_host);
  }

  const int n_teams = (n_cells + cells_per_team - 1) / cells_per_team;

  MemorySpace::Default::kokkos_space::execution_space exec;
  Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
    team_policy(exec, n_teams, Kokkos::AUTO);
  team_policy.set_scratch_size(
    0, Kokkos::PerTeam(ScratchView::shmem_size(cells_per_team * n_dofs)));

  Kokkos::parallel_for(team_policy,
                       ResolveBatch<dim, fe_degree, transpose>{weights,
                                                               masks,
                                                               src,
                                                               dst_cell_by_cell,
                                                               dst_batched,
                                                               static_cast<int>(
                                                                 n_cells),
                                                               cells_per_team});
  Kokkos::fence();

  auto cell_by_cell_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst_cell_by_cell);
  auto batched_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst_batched);

  double sum = 0., sum_of_squares = 0., max_difference = 0.;
  for (unsigned int i = 0; i < n_cells * n_dofs; ++i)
    {
      sum += cell_by_cell_host(i);
      sum_of_squares += cell_by_cell_host(i) * cell_by_cell_host(i);
      max_difference =
        std::max(max_difference,
                 std::abs(batched_host(i) - cell_by_cell_host(i)));
    }

  deallog << "  checksum cell by cell: " << sum << " " << sum_of_squares
          << std::endl;
  deallog << "  max |batched - cell by cell|: " << max_difference << std::endl;

  // the smallest case in full
  if (dim == 2 && fe_degree == 1)
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        deallog << "  mask " << static_cast<unsigned int>(make_mask(cell))
                << ":";
        for (unsigned int i = 0; i < n_dofs; ++i)
          deallog << " " << batched_host(cell * n_dofs + i);
        deallog << std::endl;
      }
}



int
main()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog << std::setprecision(12);

  Kokkos::initialize();
  {
    // cells per team chosen such that the last team has a shorter batch
    test<2, 1, false>(5);
    test<2, 1, true>(5);
    test<2, 2, false>(5);
    test<2, 2, true>(5);
    test<2, 3, false>(5);
    test<2, 3, true>(5);

    test<3, 1, false>(7);
    test<3, 1, true>(7);
    test<3, 2, false>(7);
    test<3, 2, true>(7);
    test<3, 3, false>(7);
    test<3, 3, true>(7);

    // a single cell per team, how the cell-wise callers would use the kernel
    test<2, 1, false>(1);
    test<2, 1, true>(1);
    test<2, 2, false>(1);
    test<2, 2, true>(1);
    test<2, 3, false>(1);
    test<2, 3, true>(1);

    test<3, 1, false>(1);
    test<3, 1, true>(1);
    test<3, 2, false>(1);
    test<3, 2, true>(1);
    test<3, 3, false>(1);
    test<3, 3, true>(1);
  }
  Kokkos::finalize();

  return 0;
}
