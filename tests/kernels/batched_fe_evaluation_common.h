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

#ifndef dealii_tests_batched_fe_evaluation_common_h
#define dealii_tests_batched_fe_evaluation_common_h

// Common harness for the tests of
// Copy::Portable::internal::batched::FEEvaluation and
// Copy::Portable::internal::batched::cell_loop().
//
// The cells of each color of a Portable::MatrixFree object are processed in
// batches of batch_size cells (the last batch of a color may be shorter) by
// at most two teams: a team sets up its shared memory once and then handles
// several batches one after the other. For each batch: read_dof_values(),
// evaluate(), an operator at the quadrature points of all cells of the
// batch, integrate() and distribute_local_to_global(). The operator uses a
// coefficient stored per cell and quadrature point in global memory. The
// result is compared with the same operator integrated cell by cell with
// FEValues, with the constraints (zero boundary values and hanging nodes)
// applied through AffineConstraints.
//
// An operator is a struct with
//  - static constexpr int n_components;
//  - static void apply(fe_eval, q_point, coefficient) on the device;
//  - static void reference(fe_values, q, coefficient, u, local_dst), which
//    adds the contribution of quadrature point q for the local dof values u.
// or, for a gradient-only operator applied through evaluate_and_apply(),
// instead of apply():
//  - static constexpr bool fused = true;
//  - static gradient_type flux(gradient, coefficient) on the device.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <memory>
#include <vector>

#include "../tests.h"

#include "matrix_free_dealii/portable_batched_cell_loop.h"
#include "matrix_free_dealii/portable_batched_fe_evaluation.h"


using TeamHandle = Kokkos::TeamPolicy<
  MemorySpace::Default::kokkos_space::execution_space>::member_type;



// Unit cube with a mix of Cartesian and general cells (the vertices with
// x > 0.5 are moved), the same with two levels of local refinement and hence
// hanging nodes, or sheared to a parallelepiped with affine cells only.
enum class Mesh
{
  deformed,
  hanging_nodes,
  sheared
};

template <int dim>
void
make_triangulation(Triangulation<dim> &tria, const Mesh mesh)
{
  GridGenerator::hyper_cube(tria);
  if (mesh == Mesh::hanging_nodes)
    {
      tria.refine_global(dim == 2 ? 2 : 1);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->center()[0] < 0.5)
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->center()[0] < 0.25 && cell->center()[1] < 0.5)
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }
  else
    tria.refine_global(dim == 2 ? 3 : 2);

  if (mesh == Mesh::sheared)
    GridTools::transform(
      [](const Point<dim> &p) {
        Point<dim> result = p;
        result[0] += 0.3 * p[1];
        return result;
      },
      tria);
  else
    GridTools::transform(
      [](const Point<dim> &p) {
        Point<dim> result = p;
        if (p[0] > 0.5)
          result[1] += 0.8 * (p[0] - 0.5) * p[1] * (1. - p[1]);
        return result;
      },
      tria);
}



template <int dim>
double
coefficient_value(const Point<dim> &p)
{
  double result = 1.;
  for (unsigned int d = 0; d < dim; ++d)
    result += (d + 1.) * p[d] * p[d];
  return result;
}



template <typename Operator, typename = void>
struct is_fused : std::false_type
{};

template <typename Operator>
struct is_fused<Operator, std::void_t<decltype(Operator::fused)>>
  : std::bool_constant<Operator::fused>
{};



// The operator on the cells of one batch, with the signature expected by
// Copy::Portable::internal::batched::cell_loop().
template <int dim, int fe_degree, int n_q_points_1d, typename Operator>
struct BatchedLocalOperator
{
  using Number      = double;
  using BatchedData = Copy::Portable::internal::batched::BatchedData<dim, Number>;

  static constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const BatchedData                    *data,
             const Portable::DeviceVector<Number> &src,
             Portable::DeviceVector<Number>       &dst) const
  {
    Copy::Portable::internal::batched::FEEvaluation<dim,
                                              fe_degree,
                                              n_q_points_1d,
                                              Operator::n_components,
                                              Number>
      fe_eval(data);

    // position of a point in a field with n_q_points entries per cell
    const auto position = [&](const int q_point) {
      return fe_eval.get_current_cell_index(q_point) * n_q_points +
             q_point % n_q_points;
    };

    fe_eval.read_dof_values(src);

    if constexpr (is_fused<Operator>::value)
      {
        fe_eval.evaluate_and_apply(
          EvaluationFlags::gradients,
          [&](const auto &gradient, const int q_point) {
            return Operator::flux(gradient, coefficient(position(q_point)));
          });
        fe_eval.integrate(EvaluationFlags::gradients);
      }
    else
      {
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        data->for_each_quad_point([&](const int &q_point) {
          Operator::apply(fe_eval, q_point, coefficient(position(q_point)));
        });
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      }

    fe_eval.distribute_local_to_global(dst);
  }

  Portable::DeviceVector<Number> coefficient;
};



template <int dim, int fe_degree, int n_q_points_1d, typename Operator>
struct BatchedCellOperator
{
  using Number = double;
  using PrecomputedData =
    typename Portable::MatrixFree<dim, Number>::PrecomputedData;
  using SharedData  = Copy::Portable::internal::batched::SharedData<dim, Number>;
  using BatchedData = Copy::Portable::internal::batched::BatchedData<dim, Number>;

  static constexpr int n_q_points_per_cell = Utilities::pow(n_q_points_1d, dim);

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

        Portable::DeviceVector<Number> dst_nonconst = dst;
        BatchedLocalOperator<dim, fe_degree, n_q_points_1d, Operator>{
          coefficient}(&data, src, dst_nonconst);
      }
  }

  PrecomputedData                gpu_data;
  int                            batch_size;
  Portable::DeviceVector<Number> coefficient;
  Portable::DeviceVector<Number> src;
  Portable::DeviceVector<Number> dst;
};



template <int dim, int fe_degree, int n_q_points_1d, typename Operator>
void
test(const bool lobatto,
     const Mesh mesh,
     const bool use_coloring,
     const int  batch_size,
     const bool use_cell_loop = false)
{
  using Number  = double;
  using Functor = BatchedCellOperator<dim, fe_degree, n_q_points_1d, Operator>;
  using GeometryType = ::dealii::internal::MatrixFreeFunctions::GeometryType;
  using ConstraintKinds =
    ::dealii::internal::MatrixFreeFunctions::ConstraintKinds;

  constexpr unsigned int n_q_points_per_cell =
    Utilities::pow(n_q_points_1d, dim);

  Triangulation<dim> tria;
  make_triangulation(tria, mesh);

  const FE_Q<dim>     fe_q(fe_degree);
  const FESystem<dim> fe(fe_q, Operator::n_components);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  const MappingQ1<dim> mapping;

  std::unique_ptr<Quadrature<1>> quadrature_1d;
  if (lobatto)
    quadrature_1d = std::make_unique<QGaussLobatto<1>>(n_q_points_1d);
  else
    quadrature_1d = std::make_unique<QGauss<1>>(n_q_points_1d);
  const Quadrature<dim> quadrature(*quadrature_1d);

  Portable::MatrixFree<dim, Number>                          matrix_free;
  typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values;
  additional_data.use_coloring = use_coloring;
  matrix_free.reinit(
    mapping, dof_handler, constraints, *quadrature_1d, additional_data);

  const unsigned int n_dofs   = dof_handler.n_dofs();
  const unsigned int n_colors = matrix_free.get_colored_graph().size();

  // Random source vector, also on the constrained dofs, whose entries must
  // not be used, and the function it represents
  Vector<double> src_host(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    src_host(i) = static_cast<double>(Testing::rand()) / RAND_MAX;
  Vector<double> src_distributed(src_host);
  constraints.distribute(src_distributed);

  // Reference with FEValues, coefficient per cell and quadrature point in
  // the cell numbering of Portable::MatrixFree, the number of cells of each
  // geometry type and the number of cells with hanging nodes
  Vector<double>              reference(n_dofs);
  std::vector<double>         coefficient_host(tria.n_active_cells() *
                                       n_q_points_per_cell);
  std::array<unsigned int, 3> n_cells_of_type{{0, 0, 0}};
  unsigned int                n_cells_with_hanging_nodes = 0;
  {
    FEValues<dim>                        fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_gradients |
                              update_JxW_values | update_quadrature_points);
    const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    Vector<double>                       u(dofs_per_cell);
    Vector<double>                       local_dst(dofs_per_cell);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const auto gpu_data      = matrix_free.get_data(color);
        const auto gpu_data_host = Portable::copy_mf_data_to_host<dim, Number>(
          gpu_data, additional_data.mapping_update_flags);

        for (unsigned int cell_id = 0; cell_id < gpu_data.n_cells; ++cell_id)
          {
            const GeometryType type = gpu_data_host.cell_type(cell_id);
            ++n_cells_of_type[type == GeometryType::cartesian ?
                                0 :
                                (type == GeometryType::affine ? 1 : 2)];
            if (gpu_data_host.constraint_mask(cell_id *
                                              Operator::n_components) !=
                ConstraintKinds::unconstrained)
              ++n_cells_with_hanging_nodes;

            const auto cell = matrix_free.get_cell_iterator(color, cell_id);
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              u(i) = src_distributed(dof_indices[i]);

            local_dst = 0.;
            for (unsigned int q = 0; q < n_q_points_per_cell; ++q)
              {
                const double coefficient =
                  coefficient_value(fe_values.quadrature_point(q));
                coefficient_host[(gpu_data.first_cell + cell_id) *
                                   n_q_points_per_cell +
                                 q] = coefficient;
                Operator::reference(fe_values, q, coefficient, u, local_dst);
              }

            constraints.distribute_local_to_global(local_dst,
                                                   dof_indices,
                                                   reference);
          }
      }
  }

  // Device vectors
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src(n_dofs);
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dst(n_dofs);
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> coefficient(
    coefficient_host.size());
  {
    LinearAlgebra::ReadWriteVector<Number> rw(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      rw[i] = src_host(i);
    src.import_elements(rw, VectorOperation::insert);

    LinearAlgebra::ReadWriteVector<Number> rw_coefficient(
      coefficient_host.size());
    for (unsigned int i = 0; i < coefficient_host.size(); ++i)
      rw_coefficient[i] = coefficient_host[i];
    coefficient.import_elements(rw_coefficient, VectorOperation::insert);
  }
  dst = 0.;

  // Batched cell loop, with the launch of the library (batch_size < 0:
  // automatic batch size) or with at most two teams per color
  if (use_cell_loop)
    Copy::Portable::internal::batched::cell_loop(
      matrix_free,
      BatchedLocalOperator<dim, fe_degree, n_q_points_1d, Operator>{
        Portable::DeviceVector<Number>(coefficient.get_values(),
                                       coefficient.size())},
      src,
      dst,
      batch_size < 0 ? numbers::invalid_unsigned_int : batch_size);
  else
    {
      MemorySpace::Default::kokkos_space::execution_space exec;
      for (unsigned int color = 0; color < n_colors; ++color)
        {
          const auto gpu_data = matrix_free.get_data(color);
          if (gpu_data.n_cells == 0)
            continue;

          // Fewer teams than batches, so that a team handles several batches
          const int n_batches =
            (gpu_data.n_cells + batch_size - 1) / batch_size;
          const int n_teams = std::min(n_batches, 2);

          Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>
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
                    Portable::DeviceVector<Number>(coefficient.get_values(),
                                                   coefficient.size()),
                    Portable::DeviceVector<Number>(src.get_values(), n_dofs),
                    Portable::DeviceVector<Number>(dst.get_values(), n_dofs)});
          Kokkos::fence();
        }
    }

  LinearAlgebra::ReadWriteVector<Number> dst_host(n_dofs);
  dst_host.import_elements(dst, VectorOperation::insert);

  double error = 0., norm = 0.;
  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      error = std::max(error, std::abs(dst_host[i] - reference(i)));
      norm  = std::max(norm, std::abs(reference(i)));
    }
  const double relative = error / norm;

  deallog << "dim=" << dim << " degree=" << fe_degree
          << " n_q_points_1d=" << n_q_points_1d
          << (lobatto ? " lobatto" : " gauss")
          << (mesh == Mesh::deformed ?
                " deformed" :
                (mesh == Mesh::sheared ? " sheared" : " hanging_nodes"))
          << " colors=" << n_colors << " batch_size=";
  if (batch_size < 0)
    deallog << "auto";
  else
    deallog << batch_size;
  deallog << " cells: " << n_cells_of_type[0] << " cartesian, "
          << n_cells_of_type[1] << " affine, " << n_cells_of_type[2]
          << " general, " << n_cells_with_hanging_nodes
          << " with hanging nodes, error " << (relative < 1e-12 ? 0. : relative)
          << std::endl;
}



// All meshes, with and without coloring, and several batch sizes for one
// element and quadrature formula.
template <int dim, int fe_degree, int n_q_points_1d, typename Operator>
void
run(const bool lobatto = false)
{
  for (const Mesh mesh : {Mesh::deformed, Mesh::hanging_nodes, Mesh::sheared})
    for (const bool use_coloring : {false, true})
      for (const int batch_size : {1, 3, 8})
        test<dim, fe_degree, n_q_points_1d, Operator>(lobatto,
                                                      mesh,
                                                      use_coloring,
                                                      batch_size);
}

#endif
