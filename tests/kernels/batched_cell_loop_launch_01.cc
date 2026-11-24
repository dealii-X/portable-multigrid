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

// Test the launch parameters of a batched cell loop,
// Copy::Portable::internal::batched::compute_batch_size() and make_team_policy():
// the automatic batch size is the largest one whose cell data fits
// cell_data_bytes_per_batch, the team scratch memory fits the limit of the
// backend and is enough for SharedData::reinit(), and teams that handle the
// batches league_rank(), league_rank() + league_size(), ... visit each cell
// of each color exactly once. Team and league sizes and the limit of the
// team scratch memory depend on the backend and are only checked, not
// printed.

#include "batched_fe_evaluation_common.h"


template <int dim>
struct CountCells
{
  using Number = double;
  using PrecomputedData =
    typename Portable::MatrixFree<dim, Number>::PrecomputedData;
  using SharedData = Copy::Portable::internal::batched::SharedData<dim, Number>;

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    Kokkos::Array<SharedData, Portable::n_max_dof_handlers> shared_data;
    for (unsigned int d = 0; d < n_dof_handler; ++d)
      shared_data[d].reinit(team_member,
                            n_q_points_per_cell,
                            batch_size,
                            precomputed_data[d]);
    team_member.team_barrier();

    // The team scratch memory must suffice for all allocations
    bool shared_memory_ok = true;
    for (unsigned int d = 0; d < n_dof_handler; ++d)
      shared_memory_ok = shared_memory_ok &&
                         shared_data[d].values.data() != nullptr &&
                         shared_data[d].gradients.data() != nullptr &&
                         shared_data[d].scratch_pad.data() != nullptr;

    const int n_cells = precomputed_data[0].n_cells;
    for (int batch = team_member.league_rank(); batch * batch_size < n_cells;
         batch += team_member.league_size())
      {
        const int first = batch * batch_size;
        const int last =
          (first + batch_size < n_cells) ? first + batch_size : n_cells;

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, last - first),
                             [&](const int &b) {
                               Kokkos::atomic_add(
                                 &n_visits(precomputed_data[0].first_cell +
                                           first + b),
                                 shared_memory_ok ? 1 : 1000);
                             });
        team_member.team_barrier();
      }
  }

  Kokkos::Array<PrecomputedData, Portable::n_max_dof_handlers> precomputed_data;
  unsigned int                                                 n_dof_handler;
  unsigned int                                            n_q_points_per_cell;
  int                                                     batch_size;
  Kokkos::View<int *, MemorySpace::Default::kokkos_space> n_visits;
};



template <int dim, int fe_degree, int n_q_points_1d, int n_components>
void
test(const bool use_coloring)
{
  using Number         = double;
  using ExecutionSpace = MemorySpace::Default::kokkos_space::execution_space;
  using TeamPolicy     = Kokkos::TeamPolicy<ExecutionSpace>;
  namespace batched    = Copy::Portable::internal::batched;

  const unsigned int n_q_points_per_cell = Utilities::pow(n_q_points_1d, dim);

  Triangulation<dim> tria;
  make_triangulation(tria, Mesh::deformed);

  const FE_Q<dim>     fe_q(fe_degree);
  const FESystem<dim> fe(fe_q, n_components);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.close();

  const MappingQ1<dim> mapping;
  const QGauss<1>      quadrature_1d(n_q_points_1d);

  Portable::MatrixFree<dim, Number>                          matrix_free;
  typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values;
  additional_data.use_coloring = use_coloring;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature_1d, additional_data);

  const unsigned int n_cells  = tria.n_active_cells();
  const unsigned int n_colors = matrix_free.get_colored_graph().size();

  ExecutionSpace    exec;
  const std::size_t scratch_size_max = TeamPolicy::scratch_size_max(0);

  for (const unsigned int requested_batch_size :
       {numbers::invalid_unsigned_int, 1u, 7u, n_cells + 5})
    for (const unsigned int requested_team_size :
         {numbers::invalid_unsigned_int, 1u})
      {
        Kokkos::View<int *, MemorySpace::Default::kokkos_space> n_visits(
          "n_visits", n_cells);

        bool ok                = true;
        int  batch_size_color0 = 0;
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            Kokkos::Array<
              typename Portable::MatrixFree<dim, Number>::PrecomputedData,
              Portable::n_max_dof_handlers>
              precomputed_data;
            precomputed_data[0] = matrix_free.get_data(color);
            if (precomputed_data[0].n_cells == 0)
              continue;

            const int batch_size = batched::compute_batch_size<dim, Number>(
              n_q_points_per_cell, 1, precomputed_data, requested_batch_size);
            if (color == 0)
              batch_size_color0 = batch_size;

            const std::size_t scratch_size = batched::shmem_size<dim, Number>(
              n_q_points_per_cell, batch_size, 1, precomputed_data);
            ok = ok && batch_size >= 1 &&
                 (batch_size == 1 || scratch_size <= scratch_size_max);

            if (requested_batch_size == numbers::invalid_unsigned_int)
              {
                // values, gradients and scratch pad of one cell
                const std::size_t cell_data_bytes =
                  (n_components * (1 + dim) * n_q_points_per_cell +
                   precomputed_data[0].scratch_pad_size) *
                  sizeof(Number);
                ok = ok &&
                     (batch_size == 1 || batch_size * cell_data_bytes <=
                                           batched::cell_data_bytes_per_batch);
                // not a smaller batch than necessary
                if (batched::shmem_size<dim, Number>(n_q_points_per_cell,
                                                     batch_size + 1,
                                                     1,
                                                     precomputed_data) <=
                    scratch_size_max)
                  ok = ok && (batch_size + 1) * cell_data_bytes >
                               batched::cell_data_bytes_per_batch;
              }
            else
              ok = ok && batch_size <= static_cast<int>(requested_batch_size);

            const CountCells<dim> functor{
              precomputed_data, 1, n_q_points_per_cell, batch_size, n_visits};

            const TeamPolicy team_policy =
              batched::make_team_policy<dim, Number>(exec,
                                                     functor,
                                                     n_q_points_per_cell,
                                                     batch_size,
                                                     1,
                                                     precomputed_data,
                                                     requested_team_size);

            const int n_batches =
              (precomputed_data[0].n_cells + batch_size - 1) / batch_size;
            ok = ok && team_policy.league_size() >= 1 &&
                 team_policy.league_size() <= n_batches &&
                 team_policy.team_size() >= 1 &&
                 team_policy.scratch_size(0) >= scratch_size;
            if (requested_team_size != numbers::invalid_unsigned_int)
              ok = ok && team_policy.team_size() <=
                           static_cast<int>(requested_team_size);

            Kokkos::parallel_for(team_policy, functor);
            Kokkos::fence();
          }

        const auto n_visits_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, n_visits);
        unsigned int n_cells_visited_once = 0;
        for (unsigned int cell = 0; cell < n_cells; ++cell)
          if (n_visits_host(cell) == 1)
            ++n_cells_visited_once;

        deallog << "dim=" << dim << " degree=" << fe_degree
                << " n_q_points_1d=" << n_q_points_1d
                << " n_components=" << n_components << " colors=" << n_colors
                << " requested_batch_size=";
        if (requested_batch_size == numbers::invalid_unsigned_int)
          deallog << "auto";
        else
          deallog << requested_batch_size;
        deallog << " requested_team_size=";
        if (requested_team_size == numbers::invalid_unsigned_int)
          deallog << "auto";
        else
          deallog << requested_team_size;
        deallog << ':';
        // a requested batch size may be reduced to a limit of the backend
        if (requested_batch_size == numbers::invalid_unsigned_int)
          deallog << " batch_size=" << batch_size_color0;
        deallog << " launch=" << (ok ? "OK" : "FAILED")
                << " cells visited once " << n_cells_visited_once << '/'
                << n_cells << std::endl;
      }
}



int
main()
{
  initlog();
  Kokkos::initialize();

  for (const bool use_coloring : {false, true})
    {
      test<2, 1, 2, 1>(use_coloring);
      test<2, 4, 5, 1>(use_coloring);
      test<3, 1, 2, 1>(use_coloring);
      test<3, 2, 3, 1>(use_coloring);
      test<3, 2, 3, 3>(use_coloring);
      // more quadrature points than dofs per cell
      test<3, 2, 4, 1>(use_coloring);
      // one cell needs more than cell_data_bytes_per_batch
      test<3, 4, 5, 3>(use_coloring);
    }

  Kokkos::finalize();
  return 0;
}
