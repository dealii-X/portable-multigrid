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

// Test get_dof_value() and submit_dof_value() of
// Copy::Portable::internal::batched::FEEvaluation, with the same launch as in
// batched_fe_evaluation_common.h: after read_dof_values(), the
// dof values of all cells of a batch are read with get_dof_value(), scaled
// by a weight that depends on the cell, the dof and the component, written
// back with submit_dof_value() and added to a vector with
// distribute_local_to_global(). Both the values read and the resulting
// vector are compared with the dof values obtained from the dof indices of
// the cells, with the dofs of a component in lexicographic order.

#include <deal.II/fe/fe_tools.h>

#include "batched_fe_evaluation_common.h"


DEAL_II_HOST_DEVICE inline double
weight(const int cell, const int dof, const int component)
{
  return 1. + 0.5 * cell + 0.25 * dof + 0.125 * component;
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components>
struct BatchedDoFValues
{
  using Number = double;
  using PrecomputedData =
    typename Portable::MatrixFree<dim, Number>::PrecomputedData;
  using SharedData  = Copy::Portable::internal::batched::SharedData<dim, Number>;
  using BatchedData = Copy::Portable::internal::batched::BatchedData<dim, Number>;

  static constexpr int n_q_points_per_cell = Utilities::pow(n_q_points_1d, dim);
  static constexpr int n_dofs              = Utilities::pow(fe_degree + 1, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    Kokkos::Array<PrecomputedData, Portable::n_max_dof_handlers>
      precomputed_data;
    precomputed_data[0] = gpu_data;

    // Shared memory for a full batch, set up once per team
    Kokkos::Array<SharedData, Portable::n_max_dof_handlers> shared_data;
    shared_data[0].reinit(team_member,
                          n_q_points_per_cell,
                          batch_size,
                          precomputed_data[0]);
    // reinit() stages the shape data without a trailing barrier
    team_member.team_barrier();

    // Each team handles every league_size()-th batch of the color
    const int n_cells = gpu_data.n_cells;
    for (int batch = team_member.league_rank(); batch * batch_size < n_cells;
         batch += team_member.league_size())
      {
        // The cells of the batch, numbered within the current color
        const int                    first = batch * batch_size;
        const Kokkos::pair<int, int> cell_range(
          first, (first + batch_size < n_cells) ? first + batch_size : n_cells);

        const BatchedData data{team_member,
                               n_q_points_per_cell,
                               1,
                               cell_range,
                               precomputed_data,
                               shared_data};

        Copy::Portable::internal::batched::
          FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
            fe_eval(&data);

        fe_eval.read_dof_values(src);

        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team_member, data.batch_size() * n_dofs),
          [&](const int &dof_index) {
            // cell in the numbering over all colors
            const int cell = static_cast<int>(gpu_data.first_cell) +
                             cell_range.first + dof_index / n_dofs;
            const int dof = dof_index % n_dofs;

            auto value = fe_eval.get_dof_value(dof_index);
            if constexpr (n_components == 1)
              {
                values_read(cell * n_dofs + dof, 0) = value;
                value *= weight(cell, dof, 0);
              }
            else
              {
                for (int c = 0; c < n_components; ++c)
                  {
                    values_read(cell * n_dofs + dof, c) = value[c];
                    value[c] *= weight(cell, dof, c);
                  }
              }
            fe_eval.submit_dof_value(value, dof_index);
          });
        team_member.team_barrier();

        Portable::DeviceVector<Number> dst_nonconst = dst;
        fe_eval.distribute_local_to_global(dst_nonconst);
      }
  }

  PrecomputedData                                             gpu_data;
  int                                                         batch_size;
  Kokkos::View<Number **, MemorySpace::Default::kokkos_space> values_read;
  Portable::DeviceVector<Number>                              src;
  Portable::DeviceVector<Number>                              dst;
};



template <int dim, int fe_degree, int n_q_points_1d, int n_components>
void
test(const bool use_coloring, const int batch_size)
{
  using Number  = double;
  using Functor = BatchedDoFValues<dim, fe_degree, n_q_points_1d, n_components>;

  constexpr unsigned int n_q_points_per_cell =
    Utilities::pow(n_q_points_1d, dim);
  constexpr unsigned int n_dofs_per_component =
    Utilities::pow(fe_degree + 1, dim);

  Triangulation<dim> tria;
  make_triangulation(tria, Mesh::deformed);

  const FE_Q<dim>     fe_q(fe_degree);
  const FESystem<dim> fe(fe_q, n_components);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
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

  const unsigned int n_dofs   = dof_handler.n_dofs();
  const unsigned int n_cells  = tria.n_active_cells();
  const unsigned int n_colors = matrix_free.get_colored_graph().size();

  // Random source vector, also on the constrained dofs, which must be read
  // as zero
  Vector<double> src_host(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    src_host(i) = static_cast<double>(Testing::rand()) / RAND_MAX;

  // Reference from the dof indices of the cells
  const std::vector<unsigned int> lexicographic =
    FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

  Vector<double>      reference_dst(n_dofs);
  std::vector<double> reference_read(n_cells * n_dofs_per_component *
                                     n_components);
  {
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const auto gpu_data = matrix_free.get_data(color);
        for (unsigned int cell_id = 0; cell_id < gpu_data.n_cells; ++cell_id)
          {
            const unsigned int cell_index = gpu_data.first_cell + cell_id;
            const auto cell = matrix_free.get_cell_iterator(color, cell_id);
            cell->get_dof_indices(dof_indices);

            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int j = 0; j < n_dofs_per_component; ++j)
                {
                  const types::global_dof_index dof =
                    dof_indices[fe.component_to_system_index(c,
                                                             lexicographic[j])];
                  const double value =
                    constraints.is_constrained(dof) ? 0. : src_host(dof);

                  reference_read[(cell_index * n_dofs_per_component + j) *
                                   n_components +
                                 c] = value;
                  if (!constraints.is_constrained(dof))
                    reference_dst(dof) += weight(cell_index, j, c) * value;
                }
          }
      }
  }

  // Device data
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src(n_dofs);
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst(n_dofs);
  {
    LinearAlgebra::ReadWriteVector<Number> rw(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      rw[i] = src_host(i);
    src.import_elements(rw, VectorOperation::insert);
  }
  dst = 0.;

  Kokkos::View<Number **, MemorySpace::Default::kokkos_space> values_read(
    "values_read", n_cells * n_dofs_per_component, n_components);

  // Batched cell loop
  MemorySpace::Default::kokkos_space::execution_space exec;
  for (unsigned int color = 0; color < n_colors; ++color)
    {
      const auto gpu_data = matrix_free.get_data(color);
      if (gpu_data.n_cells == 0)
        continue;

      // Fewer teams than batches, so that a team handles several batches
      const int n_batches = (gpu_data.n_cells + batch_size - 1) / batch_size;
      const int n_teams   = std::min(n_batches, 2);

      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
        team_policy(exec, n_teams, Kokkos::AUTO);
      team_policy.set_scratch_size(
        0,
        Kokkos::PerTeam(Functor::SharedData::shmem_size(n_q_points_per_cell,
                                                        batch_size,
                                                        gpu_data)));

      Kokkos::parallel_for(
        team_policy,
        Functor{gpu_data,
                batch_size,
                values_read,
                Portable::DeviceVector<Number>(src.get_values(), n_dofs),
                Portable::DeviceVector<Number>(dst.get_values(), n_dofs)});
      Kokkos::fence();
    }

  // Compare
  const auto values_read_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values_read);
  double error_read = 0., norm_read = 0.;
  for (unsigned int i = 0; i < n_cells * n_dofs_per_component; ++i)
    for (unsigned int c = 0; c < n_components; ++c)
      {
        const double reference = reference_read[i * n_components + c];
        error_read =
          std::max(error_read, std::abs(values_read_host(i, c) - reference));
        norm_read = std::max(norm_read, std::abs(reference));
      }

  LinearAlgebra::ReadWriteVector<Number> dst_host(n_dofs);
  dst_host.import_elements(dst, VectorOperation::insert);
  double error_write = 0., norm_write = 0.;
  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      error_write =
        std::max(error_write, std::abs(dst_host[i] - reference_dst(i)));
      norm_write = std::max(norm_write, std::abs(reference_dst(i)));
    }

  const auto rounded = [](const double error, const double norm) {
    const double relative = error / norm;
    return relative < 1e-12 ? 0. : relative;
  };

  deallog << "dim=" << dim << " degree=" << fe_degree
          << " n_q_points_1d=" << n_q_points_1d
          << " n_components=" << n_components << " colors=" << n_colors
          << " batch_size=" << batch_size << ": get_dof_value "
          << rounded(error_read, norm_read) << ", submit_dof_value "
          << rounded(error_write, norm_write) << std::endl;
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components>
void
run_dof_values()
{
  for (const bool use_coloring : {false, true})
    for (const int batch_size : {1, 3, 8})
      test<dim, fe_degree, n_q_points_1d, n_components>(use_coloring,
                                                        batch_size);
}



int
main()
{
  initlog();
  Kokkos::initialize();

  run_dof_values<2, 2, 3, 1>();
  // more quadrature points than dofs per cell
  run_dof_values<2, 2, 4, 1>();
  run_dof_values<2, 1, 2, 2>();
  run_dof_values<3, 1, 2, 1>();
  run_dof_values<3, 2, 3, 3>();

  Kokkos::finalize();
  return 0;
}
