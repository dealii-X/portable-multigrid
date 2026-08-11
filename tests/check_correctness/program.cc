// Correctness check for BK3Block::Parallel::KokkosKernelBlock
// (include/kernels/bk3_kokkos_kernel_block.h) against K sequential calls
// of the original BK3::Parallel::KokkosKernel (include/kernels/bk3_kokkos_kernel.h).
//
// Builds a small Portable::MatrixFree Laplace setup (mirroring
// include/operators/portable_laplace_operator_bk3.h's constructor, duplicated
// here rather than modifying that header), applies both kernels to the same
// K random right-hand sides, and compares the results column by column.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include "kernels/bk3_kokkos_kernel.h"
#include "kernels/bk3_kokkos_kernel_block.h"

using namespace dealii;

constexpr int dim       = 3;
constexpr int fe_degree = 4;
using Number             = double;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int n_rhs = 11; // K -- mirrors n_local_coarse_dofs (7-17 in practice)

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(4);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  constraints.close();

  const MappingQ<dim> mapping(fe_degree);
  const QGauss<1>      quadrature_1d(fe_degree + 1);

  Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;

  Portable::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

  // --- setup dof_indices_per_color / G_tensors, mirroring
  //     Portable::LaplaceOperatorBK3 exactly (not reused directly since
  //     those members are private there) ---
  constexpr unsigned int n_local_dofs  = Utilities::pow(fe_degree + 1, dim);
  constexpr unsigned int n_q_points    = Utilities::pow(fe_degree + 1, dim);
  constexpr int          symmetric_dim = (dim * (dim + 1)) / 2;

  std::vector<unsigned int> lex_numbering(n_local_dofs);
  {
    const Quadrature<1>                                      dummy_quadrature(
      std::vector<Point<1>>(1, Point<1>()));
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
    lex_numbering = shape_info.lexicographic_numbering;
  }

  const auto        &colored_graph = matrix_free.get_colored_graph();
  const unsigned int n_colors      = colored_graph.size();

  std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
    dof_indices_per_color(n_colors);
  std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> G_tensors(n_colors);

  std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);

  for (unsigned int color = 0; color < n_colors; ++color)
    {
      const unsigned int n_cells = colored_graph[color].size();
      if (n_cells == 0)
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
          typename DoFHandler<dim>::cell_iterator cell = graph[cell_id];
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < n_local_dofs; ++i)
            {
              const auto global_dof = local_dof_indices[lex_numbering[i]];
              if (constraints.is_constrained(global_dof))
                dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
              else
                dof_indices_host(i, cell_id) = global_dof;
            }
        }

      Kokkos::deep_copy(dof_indices_per_color[color], dof_indices_host);

      G_tensors[color] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("G_tensor_color_" + std::to_string(color), Kokkos::WithoutInitializing),
        symmetric_dim * mf_data.n_cells * n_q_points);
      auto G = G_tensors[color];

      const auto &inv_jacobian = mf_data.inv_jacobian;
      const auto &JxW          = mf_data.JxW;
      const unsigned int color_n_cells = mf_data.n_cells;

      Kokkos::parallel_for(
        "Fill_G_tensor_color" + std::to_string(color),
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0,
                                                                                  color_n_cells),
        KOKKOS_LAMBDA(const int cell_id) {
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              Number components[symmetric_dim];
              int    idx = 0;
              for (int d1 = 0; d1 < dim; ++d1)
                for (int d2 = d1; d2 < dim; ++d2)
                  {
                    Number sum = 0;
                    for (int k = 0; k < dim; ++k)
                      sum += inv_jacobian(q_point, cell_id, d1, k) *
                             inv_jacobian(q_point, cell_id, d2, k);
                    components[idx] = JxW(q_point, cell_id) * sum;
                    ++idx;
                  }
              for (int c = 0; c < symmetric_dim; ++c)
                G[cell_id * symmetric_dim * n_q_points + c * n_q_points + q_point] = components[c];
            }
        });
      Kokkos::fence();
    }

  // --- build K random inputs, both as separate vectors (for the reference
  //     sequential-kernel path) and packed into one n_rhs*dof_stride block
  //     (for the block kernel) ---
  const unsigned int dof_stride = dof_handler.n_dofs();

  std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> x(n_rhs), dst_ref(n_rhs);

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> x_block("x_block",
                                                                       n_rhs * dof_stride);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> dst_block("dst_block",
                                                                         n_rhs * dof_stride);

  std::mt19937                          rng(42);
  std::uniform_real_distribution<Number> dist(-1., 1.);

  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      x[k]       = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>("x_" +
                                                                             std::to_string(k),
                                                                           dof_stride);
      dst_ref[k] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>("dst_ref_" +
                                                                              std::to_string(k),
                                                                            dof_stride);

      auto x_host = Kokkos::create_mirror_view(x[k]);
      auto x_block_host_slice =
        Kokkos::create_mirror_view(Kokkos::subview(
          x_block, Kokkos::make_pair(k * dof_stride, (k + 1) * dof_stride)));

      for (unsigned int i = 0; i < dof_stride; ++i)
        {
          const Number v         = dist(rng);
          x_host(i)              = v;
          x_block_host_slice(i) = v;
        }

      Kokkos::deep_copy(x[k], x_host);
      Kokkos::deep_copy(Kokkos::subview(x_block,
                                        Kokkos::make_pair(k * dof_stride, (k + 1) * dof_stride)),
                        x_block_host_slice);

      Kokkos::deep_copy(dst_ref[k], 0.);
    }
  Kokkos::deep_copy(dst_block, 0.);
  Kokkos::fence();

  constexpr bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
  unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

  // --- reference: K sequential calls of the original kernel ---
  for (unsigned int color = 0; color < n_colors; ++color)
    {
      const unsigned int n_cells = colored_graph[color].size();
      if (n_cells == 0)
        continue;
      const auto &precomputed_data = matrix_free.get_data(color);

      for (unsigned int k = 0; k < n_rhs; ++k)
        BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
          precomputed_data.shape_values,
          precomputed_data.co_shape_gradients,
          G_tensors[color],
          x[k],
          dst_ref[k],
          dof_indices_per_color[color],
          n_cells,
          numBlocks,
          threadsPerBlock);
    }
  Kokkos::fence();

  // --- block: one call per color covering all n_rhs columns at once ---
  for (unsigned int color = 0; color < n_colors; ++color)
    {
      const unsigned int n_cells = colored_graph[color].size();
      if (n_cells == 0)
        continue;
      const auto &precomputed_data = matrix_free.get_data(color);

      BK3Block::Parallel::KokkosKernelBlock<dim, fe_degree + 1, fe_degree + 1, Number>(
        precomputed_data.shape_values,
        precomputed_data.co_shape_gradients,
        G_tensors[color],
        x_block,
        dst_block,
        dof_indices_per_color[color],
        n_cells,
        n_rhs,
        dof_stride,
        numBlocks,
        threadsPerBlock);
    }
  Kokkos::fence();

  // --- compare ---
  Number max_abs_diff = 0.;
  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      auto dst_ref_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_ref[k]);
      auto dst_block_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            Kokkos::subview(dst_block,
                                                            Kokkos::make_pair(k * dof_stride,
                                                                              (k + 1) * dof_stride)));

      Number column_max_diff = 0.;
      for (unsigned int i = 0; i < dof_stride; ++i)
        column_max_diff =
          std::max(column_max_diff, std::abs(dst_ref_host(i) - dst_block_host(i)));

      std::cout << "RHS " << k << ": max |ref - block| = " << column_max_diff << std::endl;
      max_abs_diff = std::max(max_abs_diff, column_max_diff);
    }

  std::cout << std::endl
            << "n_dofs = " << dof_stride << ", n_rhs = " << n_rhs << ", n_colors = " << n_colors
            << std::endl
            << "Overall max |ref - block| = " << max_abs_diff << std::endl;

  if (max_abs_diff > 1e-10)
    {
      std::cout << "FAILED: block kernel does not match sequential reference." << std::endl;
      return 1;
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}
