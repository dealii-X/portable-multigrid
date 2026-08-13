// Correctness check for Portable::SolverBlockCG
// (include/domain_decomposition/portable_solver_block_cg.h) against K
// sequential fixed-iteration-count calls of plain dealii::SolverCG -- the
// solver it's actually modeled on (BDDCPreconditioner::vmult_fine_correction()
// uses plain SolverCG, not SolverProjectedCG, since Ahat/Chat already
// project their own output; see the class comment in
// portable_solver_block_cg.h).
//
// Builds a small Dirichlet Laplace matrix-free setup (mirroring
// include/operators/portable_laplace_operator_bk3.h, duplicated here rather
// than modifying that header -- same approach as
// tests/check_correctness/program.cc's kernel-level check), solves K
// independent systems A*x_k = b_k both ways, and compares column by column.

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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include "domain_decomposition/portable_solver_block_cg.h"
#include "kernels/bk3_kokkos_kernel.h"
#include "kernels/bk3_kokkos_kernel_block.h"

using namespace dealii;

constexpr int dim       = 3;
constexpr int fe_degree = 3;
using Number             = double;
using VectorType         = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int n_rhs        = 9; // K
  const unsigned int n_iterations = 12;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(3);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Zero Dirichlet everywhere -> SPD operator, well-posed CG (rather than
  // the pure-Neumann, singular operator we'd get with no constraints at
  // all -- irrelevant for the earlier leaf-kernel check since that only
  // applied the operator once, but here we're actually solving).
  AffineConstraints<Number> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  const MappingQ<dim> mapping(fe_degree);
  const QGauss<1>      quadrature_1d(fe_degree + 1);

  Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;

  Portable::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

  constexpr unsigned int n_local_dofs  = Utilities::pow(fe_degree + 1, dim);
  constexpr unsigned int n_q_points    = Utilities::pow(fe_degree + 1, dim);
  constexpr int          symmetric_dim = (dim * (dim + 1)) / 2;

  std::vector<unsigned int> lex_numbering(n_local_dofs);
  {
    const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
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
      auto               G             = G_tensors[color];
      const auto        &inv_jacobian  = mf_data.inv_jacobian;
      const auto        &JxW           = mf_data.JxW;
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

  const unsigned int dof_stride = dof_handler.n_dofs();

  constexpr bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
  unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

  // Scalar operator: applies A to one dof_stride-sized vector via the
  // original (single-RHS) kernel.
  struct ScalarOperator
  {
    const decltype(colored_graph)                                        &graph_ref;
    Portable::MatrixFree<dim, Number>                                    &matrix_free;
    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> &G_tensors;
    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
                      &dof_indices_per_color;
    unsigned int       numBlocks, threadsPerBlock;

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0.;

      Portable::DeviceVector<Number> src_device(src.get_values(), src.size());
      Portable::DeviceVector<Number> dst_device(dst.get_values(), dst.size());

      for (unsigned int color = 0; color < graph_ref.size(); ++color)
        {
          const unsigned int n_cells = graph_ref[color].size();
          if (n_cells == 0)
            continue;
          const auto &precomputed_data = matrix_free.get_data(color);
          BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
            precomputed_data.shape_values,
            precomputed_data.co_shape_gradients,
            G_tensors[color],
            src_device,
            dst_device,
            dof_indices_per_color[color],
            n_cells,
            numBlocks,
            threadsPerBlock);
        }
      Kokkos::fence();
    }
  } scalar_operator{colored_graph, matrix_free, G_tensors, dof_indices_per_color, numBlocks, threadsPerBlock};

  // Block operator: applies A to n_rhs blocks of dof_stride at once via
  // the block kernel.
  struct BlockOperator
  {
    const decltype(colored_graph)                                        &graph_ref;
    Portable::MatrixFree<dim, Number>                                    &matrix_free;
    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> &G_tensors;
    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
                      &dof_indices_per_color;
    unsigned int       n_rhs, dof_stride, numBlocks, threadsPerBlock;

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0.;

      Portable::DeviceVector<Number> src_device(src.get_values(), src.size());
      Portable::DeviceVector<Number> dst_device(dst.get_values(), dst.size());

      for (unsigned int color = 0; color < graph_ref.size(); ++color)
        {
          const unsigned int n_cells = graph_ref[color].size();
          if (n_cells == 0)
            continue;
          const auto &precomputed_data = matrix_free.get_data(color);
          BK3Block::Parallel::KokkosKernelBlock<dim, fe_degree + 1, fe_degree + 1, Number>(
            precomputed_data.shape_values,
            precomputed_data.co_shape_gradients,
            G_tensors[color],
            src_device,
            dst_device,
            dof_indices_per_color[color],
            n_cells,
            n_rhs,
            dof_stride,
            numBlocks,
            threadsPerBlock);
        }
      Kokkos::fence();
    }
  } block_operator{colored_graph,
                     matrix_free,
                     G_tensors,
                     dof_indices_per_color,
                     n_rhs,
                     dof_stride,
                     numBlocks,
                     threadsPerBlock};

  // --- random RHS, both as K separate vectors and packed into one block ---
  std::mt19937                          rng(1234);
  std::uniform_real_distribution<Number> dist(-1., 1.);

  std::vector<VectorType> b(n_rhs), x_ref(n_rhs);
  VectorType              b_block, x_block;

  b_block.reinit(dof_stride * n_rhs);
  x_block.reinit(dof_stride * n_rhs);
  x_block = 0.;

  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      b[k].reinit(dof_stride);
      x_ref[k].reinit(dof_stride);
      x_ref[k] = 0.;

      // Zero the boundary/constrained dofs: the operator's dof_indices
      // maps them to numbers::invalid_unsigned_int and A.vmult() never
      // writes to them (dst is zeroed then only interior positions get
      // atomic-add contributions), so an RHS with nonzero garbage there
      // could never be driven to zero residual by any number of CG
      // iterations -- exactly what caused the "divergence" seen before
      // this fix, on both the scalar reference and the block solver
      // equally (a test-setup bug, not a SolverBlockCG bug).
      auto b_host        = Kokkos::create_mirror_view(
        Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(b[k].get_values(), dof_stride));
      for (unsigned int i = 0; i < dof_stride; ++i)
        b_host(i) = constraints.is_constrained(i) ? 0. : dist(rng);

      Kokkos::deep_copy(
        Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(b[k].get_values(), dof_stride),
        b_host);
      Kokkos::deep_copy(
        Kokkos::subview(Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                          b_block.get_values(), dof_stride * n_rhs),
                        Kokkos::make_pair(k * dof_stride, (k + 1) * dof_stride)),
        b_host);
    }
  Kokkos::fence();

  // --- reference: K sequential fixed-iteration-count scalar CG solves ---
  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      IterationNumberControl control(n_iterations, 0., false, false);
      SolverCG<VectorType> solver(control);
      solver.solve(scalar_operator, x_ref[k], b[k], PreconditionIdentity());
    }

  // --- block: one call covering all n_rhs columns, same fixed-iteration
  //     control as the scalar reference above (IterationNumberControl
  //     always reports success at step == n_iterations regardless of
  //     residual, so this is an apples-to-apples comparison) ---
  {
    IterationNumberControl                control(n_iterations, 0., false, false);
    Portable::SolverBlockCG<VectorType> solver(control);
    solver.solve_block(block_operator, x_block, b_block, PreconditionIdentity(), n_rhs);
  }

  // --- compare ---
  Number max_abs_diff = 0.;
  for (unsigned int k = 0; k < n_rhs; ++k)
    {
      auto x_ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                                              x_ref[k].get_values(), dof_stride));
      auto x_block_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        Kokkos::subview(Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                          x_block.get_values(), dof_stride * n_rhs),
                        Kokkos::make_pair(k * dof_stride, (k + 1) * dof_stride)));

      Number column_max_diff = 0.;
      for (unsigned int i = 0; i < dof_stride; ++i)
        column_max_diff = std::max(column_max_diff, std::abs(x_ref_host(i) - x_block_host(i)));

      std::cout << "RHS " << k << ": max |ref - block| = " << column_max_diff << std::endl;
      max_abs_diff = std::max(max_abs_diff, column_max_diff);
    }

  std::cout << std::endl
            << "n_dofs = " << dof_stride << ", n_rhs = " << n_rhs
            << ", n_iterations = " << n_iterations << ", n_colors = " << n_colors << std::endl
            << "Overall max |ref - block| = " << max_abs_diff << std::endl;

  if (max_abs_diff > 1e-8)
    {
      std::cout << "FAILED: block CG does not match sequential reference." << std::endl;
      return 1;
    }

  // --- tolerance-based convergence: SolverControl checked against the max
  //     residual norm across all n_rhs columns (rather than a fixed
  //     iteration count) -- confirm it terminates early and that every
  //     column actually ends up within tolerance, not just the worst one.
  //
  //     Uses its own, much smaller mesh than the exact-match test above:
  //     unpreconditioned CG on the 15625-dof operator above needs far
  //     more than a few hundred iterations to converge at all (confirmed
  //     by running plain scalar SolverCG on it with the same budget --
  //     it diverges too, so this isn't a block-solve-specific issue, just
  //     an unrealistically hard problem for PreconditionIdentity). The
  //     real use case converges in ~15-18 MG-preconditioned iterations;
  //     this small mesh is what lets a plain, unpreconditioned CG stand
  //     in for that within a sane iteration budget for this test. ---
  {
    Triangulation<dim> small_triangulation;
    GridGenerator::hyper_cube(small_triangulation, 0., 1.);
    small_triangulation.refine_global(1);

    DoFHandler<dim> small_dof_handler(small_triangulation);
    small_dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> small_constraints;
    DoFTools::make_zero_boundary_constraints(small_dof_handler, small_constraints);
    small_constraints.close();

    Portable::MatrixFree<dim, Number> small_matrix_free;
    small_matrix_free.reinit(
      mapping, small_dof_handler, small_constraints, quadrature_1d, additional_data);

    const auto        &small_colored_graph = small_matrix_free.get_colored_graph();
    const unsigned int small_n_colors      = small_colored_graph.size();

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      small_dof_indices_per_color(small_n_colors);
    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> small_G_tensors(
      small_n_colors);

    for (unsigned int color = 0; color < small_n_colors; ++color)
      {
        const unsigned int n_cells = small_colored_graph[color].size();
        if (n_cells == 0)
          continue;

        const auto &mf_data = small_matrix_free.get_data(color);
        const auto &graph   = small_colored_graph[color];

        small_dof_indices_per_color[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("small_dof_indices_" + std::to_string(color),
                               Kokkos::WithoutInitializing),
            n_local_dofs,
            mf_data.n_cells);
        auto dof_indices_host = Kokkos::create_mirror_view(small_dof_indices_per_color[color]);

        for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
          {
            typename DoFHandler<dim>::cell_iterator cell = graph[cell_id];
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                const auto global_dof = local_dof_indices[lex_numbering[i]];
                if (small_constraints.is_constrained(global_dof))
                  dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                else
                  dof_indices_host(i, cell_id) = global_dof;
              }
          }

        Kokkos::deep_copy(small_dof_indices_per_color[color], dof_indices_host);

        small_G_tensors[color] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("small_G_tensor_color_" + std::to_string(color),
                             Kokkos::WithoutInitializing),
          symmetric_dim * mf_data.n_cells * n_q_points);
        auto               G             = small_G_tensors[color];
        const auto        &inv_jacobian  = mf_data.inv_jacobian;
        const auto        &JxW           = mf_data.JxW;
        const unsigned int color_n_cells = mf_data.n_cells;

        Kokkos::parallel_for(
          "Fill_small_G_tensor_color" + std::to_string(color),
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
                  G[cell_id * symmetric_dim * n_q_points + c * n_q_points + q_point] =
                    components[c];
              }
          });
        Kokkos::fence();
      }

    const unsigned int small_dof_stride = small_dof_handler.n_dofs();

    ScalarOperator small_scalar_operator{small_colored_graph,
                                         small_matrix_free,
                                         small_G_tensors,
                                         small_dof_indices_per_color,
                                         numBlocks,
                                         threadsPerBlock};
    BlockOperator small_block_operator{small_colored_graph,
                                           small_matrix_free,
                                           small_G_tensors,
                                           small_dof_indices_per_color,
                                           n_rhs,
                                           small_dof_stride,
                                           numBlocks,
                                           threadsPerBlock};

    std::vector<VectorType> small_b(n_rhs);
    VectorType              small_b_block(small_dof_stride * n_rhs);

    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        small_b[k].reinit(small_dof_stride);
        auto b_host = Kokkos::create_mirror_view(Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
          small_b[k].get_values(), small_dof_stride));
        for (unsigned int i = 0; i < small_dof_stride; ++i)
          b_host(i) = small_constraints.is_constrained(i) ? 0. : dist(rng);

        Kokkos::deep_copy(
          Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(small_b[k].get_values(),
                                                                      small_dof_stride),
          b_host);
        Kokkos::deep_copy(
          Kokkos::subview(Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                            small_b_block.get_values(), small_dof_stride * n_rhs),
                          Kokkos::make_pair(k * small_dof_stride, (k + 1) * small_dof_stride)),
          b_host);
      }
    Kokkos::fence();

    const double tolerance = 1e-6 * small_b_block.l2_norm();

    // Diagnostic: does plain scalar SolverCG with the same tolerance/
    // max_steps on a single column also diverge? Isolates whether this
    // is a block-solve-specific bug or a property of unpreconditioned CG
    // on this test operator.
    {
      const double col_tolerance = 1e-6 * small_b[0].l2_norm();
      SolverControl scalar_control(300, col_tolerance);
      VectorType    x_scalar_tol(small_dof_stride);
      x_scalar_tol = 0.;
      try
        {
          SolverCG<VectorType> scalar_solver(scalar_control);
          scalar_solver.solve(small_scalar_operator, x_scalar_tol, small_b[0], PreconditionIdentity());
          std::cout << "Scalar reference (column 0) converged in " << scalar_control.last_step()
                    << " iterations" << std::endl;
        }
      catch (const std::exception &exc)
        {
          std::cout << "Scalar reference (column 0) FAILED to converge: " << exc.what()
                    << std::endl;
        }
    }

    SolverControl control(300, tolerance);

    VectorType x_tol(small_b_block.size());
    x_tol = 0.;

    Portable::SolverBlockCG<VectorType> solver(control);
    try
      {
        solver.solve_block(small_block_operator, x_tol, small_b_block, PreconditionIdentity(), n_rhs);
      }
    catch (const std::exception &exc)
      {
        std::cout << "Block solve FAILED to converge: " << exc.what() << std::endl;
        return 1;
      }

    std::cout << std::endl
              << "Tolerance-based solve converged in " << control.last_step()
              << " iterations (tolerance = " << tolerance << ")" << std::endl;

    VectorType residual(small_b_block.size());
    small_block_operator.vmult(residual, x_tol);
    residual.sadd(-1., 1., small_b_block);

    double worst_column_residual = 0.;
    for (unsigned int k = 0; k < n_rhs; ++k)
      {
        auto residual_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          Kokkos::subview(Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                            residual.get_values(), small_dof_stride * n_rhs),
                          Kokkos::make_pair(k * small_dof_stride, (k + 1) * small_dof_stride)));

        double column_residual_norm = 0.;
        for (unsigned int i = 0; i < small_dof_stride; ++i)
          column_residual_norm += residual_host(i) * residual_host(i);
        column_residual_norm = std::sqrt(column_residual_norm);

        std::cout << "  column " << k << ": residual norm = " << column_residual_norm << std::endl;

        worst_column_residual = std::max(worst_column_residual, column_residual_norm);

        if (!(column_residual_norm <= tolerance))
          {
            std::cout << "FAILED: column " << k << " residual " << column_residual_norm
                      << " exceeds tolerance " << tolerance << std::endl;
            return 1;
          }
      }

    std::cout << "Worst-column residual = " << worst_column_residual << " (tolerance = " << tolerance
              << ")" << std::endl;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
