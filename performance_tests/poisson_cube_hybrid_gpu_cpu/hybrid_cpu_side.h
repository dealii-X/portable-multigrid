#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "hybrid_benchmark.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_switch_permutation.h"
#include "hybrid_triangulations.h"
#include "operators/cpu_laplace_operator.h"
#include "portable_multigrid_solver.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * CPU levels of the hybrid multigrid on comm_world (all ranks): the levels
   * 0..switch_level of the ph-hierarchy (Q1 h-levels, then p-levels on the
   * fine mesh up to Q_top_degree), with the operators, transfers and
   * smoothers of poisson_cube_cpu. The finest CPU level is the switch level
   * s-1, which the GPU side also holds; P moves values between the two.
   *
   * GPU ranks call setup()/coarse_solve() directly, CPU-only ranks through
   * run_service_loop().
   */
  template <int dim, int top_degree>
  class CpuSide
  {
  public:
    using vcycle_number = float; // as in poisson_cube_bk3
    using full_number   = double;

    using VectorType   = LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host>;
    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Host>;

    using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorTypeMG>;
    using TopMatrixType   = Portable::LaplaceOperatorBase<dim, VectorType>;
    using TransferType    = MGTwoLevelTransfer<dim, VectorTypeMG>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;

    using MGSolverType = multigrid::MultigridSolver<dim,
                                                    top_degree,
                                                    vcycle_number,
                                                    full_number,
                                                    SmootherType,
                                                    VectorTypeMG,
                                                    VectorType,
                                                    TransferType>;

    // n_refine, fine_degree: the full hierarchy, of which the CPU side takes
    // the bottom levels
    CpuSide(const MPILayout   &layout,
            const CoarseGrid &coarse_grid,
            const unsigned int n_smooth,
            const unsigned int n_refine,
            const unsigned int fine_degree)
      : layout(layout)
      , coarse_grid(coarse_grid)
      , n_smooth(n_smooth)
      , n_refine(n_refine)
      , fine_degree(fine_degree)
    {}

    // GPU ranks: the GPU side's DoFHandler of the switch level. setup() builds
    // P (switch-level values GPU <-> CPU) from it and checks P.
    void
    set_gpu_switch_dofs(const DoFHandler<dim> *dofs)
    {
      gpu_switch_dofs = dofs;
    }

    // Collective on comm_world: build the CPU levels 0..switch_level (index
    // in the full ph-hierarchy). with_rhs: also assemble f = 1 on the top
    // level, for solve() in a CPU-only cycle.
    void
    setup(const unsigned int switch_level, const bool with_rhs = false);

    // Collective on comm_world, CPU-only cycle: CG on the top level,
    // preconditioned by the V-cycle. Returns iterations and reduction per
    // iteration.
    std::pair<unsigned int, double>
    solve();

    // Wall time of the last solve(), maximum over the ranks.
    double
    last_solve_time() const
    {
      return solve_time;
    }

    // Collective on comm_world: best times of the operators and transfers of
    // the CPU levels (entry l = level l).
    std::vector<LevelTimes>
    benchmark_levels(const unsigned int n_calls);

    // Collective on comm_world, CPU-only cycle: best times of the top-level
    // matvec in double and in the V-cycle number type.
    std::pair<double, double>
    benchmark_matvec(const unsigned int n_calls)
    {
      return time_fine_matvecs(layout.comm_world,
                               *hierarchy->solver,
                               hierarchy->dof_handlers.back().n_dofs(),
                               n_calls);
    }

    types::global_dof_index
    n_top_dofs() const
    {
      return hierarchy->dof_handlers.back().n_dofs();
    }

    // Collective on comm_world: one V-cycle (zero initial guess) on the CPU
    // levels. Input and output are the switch-level values in GPU numbering
    // in gpu_values(): right-hand side in, solution out.
    void
    coarse_solve();

    // Switch-level values in GPU numbering: the locally owned entries of the
    // GPU side's level vector, in host memory. Empty on CPU-only ranks.
    std::vector<vcycle_number> &
    gpu_values()
    {
      return gpu_switch_values;
    }

    // Rank-local timers: per CPU level (MultigridSolver's) and of P.
    void
    reset_times()
    {
      hierarchy->solver->reset_timings();
      p_times[0] = p_times[1] = 0.;
    }

    // World rank 0: CPU-level times since reset_times(), and P down/up
    // (including the float/double conversion).
    void
    print_times()
    {
      if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
        std::cout << "==== CPU side levels, rank 0, coarse solves of the last solve" << std::endl;
      hierarchy->solver->print_wall_times();
      if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
        std::cout << "P down " << p_times[0] << " P up " << p_times[1] << std::endl;
    }

    // World rank 0, CPU-only cycle: per-level times of the last solve().
    void
    print_level_times()
    {
      if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
        std::cout << "==== CPU side levels, rank 0, last solve" << std::endl;
      hierarchy->solver->print_wall_times();
    }

  private:
    // Everything one setup() builds. Members are destroyed in reverse order:
    // solver first, mesh last.
    struct Hierarchy
    {
      Hierarchy(const MPI_Comm     comm,
                const CoarseGrid &coarse_grid,
                const unsigned int mesh_refine)
        : mesh(comm, coarse_grid, mesh_refine)
      {}

      MeshLevels<dim>                                 mesh;
      std::vector<unsigned int>                       degrees; // polynomial degree per level
      MGLevelObject<DoFHandler<dim>>                  dof_handlers;
      SwitchLevelPermutation<dim>                     permutation; // P
      MGLevelObject<AffineConstraints<vcycle_number>> constraints;
      AffineConstraints<full_number>                  top_constraints; // switch level, double
      MGLevelObject<std::unique_ptr<LevelMatrixType>> matrices;
      std::unique_ptr<TopMatrixType>                  top_matrix; // switch level, double
      MGLevelObject<std::unique_ptr<TransferType>>    transfers;
      MGLevelObject<SmootherType>                     smoothers;
      VectorType                                      rhs; // f = 1 in a CPU-only cycle, else unused
      std::unique_ptr<MGSolverType>                   solver;

      // switch-level values in CPU numbering, for coarse_solve()
      std::vector<vcycle_number> cpu_values;
      VectorType                 cpu_rhs;
      VectorType                 cpu_solution;
    };

    // creates the CPU operator of the dispatched degree in `matrix`
    struct OperatorRunner
    {
      const DoFHandler<dim>                  &dof_handler;
      const AffineConstraints<vcycle_number> &constraints;
      std::unique_ptr<LevelMatrixType>       &matrix;

      template <unsigned int degree>
      void
      run()
      {
        matrix = std::make_unique<CPU::LaplaceOperator<dim, degree, vcycle_number>>(dof_handler,
                                                                                    constraints,
                                                                                    false);
      }
    };

    void
    setup_dofs(Hierarchy &h) const;

    void
    setup_operators(Hierarchy &h) const;

    void
    compute_rhs(Hierarchy &h) const;

    const MPILayout   &layout;
    const CoarseGrid coarse_grid;
    const unsigned int n_smooth;
    const unsigned int n_refine;
    const unsigned int fine_degree;

    const DoFHandler<dim> *gpu_switch_dofs = nullptr;

    std::vector<vcycle_number> gpu_switch_values;

    double p_times[2] = {0., 0.}; // P down, P up
    double solve_time = 0.;

    std::unique_ptr<Hierarchy> hierarchy;
  };



  template <int dim, int top_degree>
  void
  CpuSide<dim, top_degree>::setup(const unsigned int switch_level, const bool with_rhs)
  {
    Timer time;

    // levels 0..switch_level of the full hierarchy
    std::vector<unsigned int> degrees = ph_level_degrees(n_refine + 1, fine_degree);
    AssertThrow(switch_level < degrees.size(),
                ExcMessage("The switch level is above the fine level."));
    degrees.resize(switch_level + 1);
    AssertThrow(degrees.back() == top_degree,
                ExcMessage("The switch level does not have degree top_degree."));

    // tear down the previous setup (solver first, mesh last), then rebuild;
    // the mesh is refined up to the switch level, at most to the fine mesh
    hierarchy.reset();
    hierarchy = std::make_unique<Hierarchy>(layout.comm_world,
                                            coarse_grid,
                                            std::min(switch_level, n_refine));
    Hierarchy &h = *hierarchy;
    h.degrees    = degrees;

    print_level_partitions(h.mesh.levels, layout.comm_world, "CPU side, comm_world", std::cout);

    setup_dofs(h);

    // only if the GPU ranks provided their switch level: who owns which
    // switch-level cells on the two sides, then P and its check
    if (Utilities::MPI::max(gpu_switch_dofs != nullptr ? 1u : 0u, layout.comm_world) == 1u)
      {
        const DoFHandler<dim> &cpu_switch_dofs = h.dof_handlers[h.dof_handlers.max_level()];

        check_switch_level_ownership<dim>(layout,
                                          gpu_switch_dofs != nullptr ?
                                            &gpu_switch_dofs->get_triangulation() :
                                            nullptr,
                                          cpu_switch_dofs.get_triangulation(),
                                          std::cout);

        h.permutation.reinit(layout, gpu_switch_dofs, cpu_switch_dofs);

        // the same on all ranks (global max)
        const double p_error = check_switch_level_permutation(
          layout, h.permutation, gpu_switch_dofs, cpu_switch_dofs, std::cout);
        AssertThrow(p_error < 1e-12, ExcMessage("P moves switch-level values to wrong DoFs."));
      }

    setup_operators(h);

    // buffers for coarse_solve(), in the MatrixFree layout of the switch level
    h.top_matrix->initialize_dof_vector(h.rhs);
    if (with_rhs)
      compute_rhs(h);
    h.top_matrix->initialize_dof_vector(h.cpu_rhs);
    h.top_matrix->initialize_dof_vector(h.cpu_solution);
    h.cpu_values.resize(h.cpu_rhs.locally_owned_size());
    gpu_switch_values.resize(
      gpu_switch_dofs != nullptr ? gpu_switch_dofs->locally_owned_dofs().n_elements() : 0);

    h.solver = std::make_unique<MGSolverType>(h.top_matrix,
                                              h.dof_handlers,
                                              h.constraints,
                                              h.transfers,
                                              h.matrices,
                                              h.smoothers,
                                              h.rhs,
                                              n_smooth,
                                              n_smooth);

    const double setup_time = Utilities::MPI::max(time.wall_time(), layout.comm_world);
    if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
      {
        std::cout << "CPU levels 0.." << h.dof_handlers.max_level() << ", degree/DoFs:";
        for (unsigned int l = 0; l <= h.dof_handlers.max_level(); ++l)
          std::cout << " Q" << h.degrees[l] << '/' << h.dof_handlers[l].n_dofs();
        std::cout << std::endl << "CPU side setup: " << setup_time << " s" << std::endl;
      }
  }



  template <int dim, int top_degree>
  void
  CpuSide<dim, top_degree>::setup_dofs(Hierarchy &h) const
  {
    const unsigned int n_h_levels = h.mesh.levels.size();
    const unsigned int max_level  = h.degrees.size() - 1;

    h.dof_handlers.resize(0, max_level);
    h.constraints.resize(0, max_level);

    for (unsigned int l = 0; l <= max_level; ++l)
      {
        DoFHandler<dim> &dof_handler = h.dof_handlers[l];
        dof_handler.reinit(*h.mesh.levels[std::min(l, n_h_levels - 1)]);
        dof_handler.distribute_dofs(FE_Q<dim>(h.degrees[l]));

        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        // built in double, as in poisson_cube_cpu; kept for the top level
        AffineConstraints<full_number>  level_constraints_double;
        AffineConstraints<full_number> &constraints =
          l == max_level ? h.top_constraints : level_constraints_double;

        constraints.reinit(dof_handler.locally_owned_dofs(), relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 types::boundary_id(0),
                                                 Functions::ZeroFunction<dim, full_number>(),
                                                 constraints);
        constraints.close();

        h.constraints[l].reinit(dof_handler.locally_owned_dofs(), relevant_dofs);
        h.constraints[l].copy_from(constraints);
        h.constraints[l].close();
      }
  }



  template <int dim, int top_degree>
  void
  CpuSide<dim, top_degree>::setup_operators(Hierarchy &h) const
  {
    const unsigned int max_level  = h.dof_handlers.max_level();
    const unsigned int n_h_levels = h.mesh.levels.size();

    // level operators in vcycle_number, the top level also in double for
    // MultigridSolver
    h.matrices.resize(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        OperatorRunner runner{h.dof_handlers[l], h.constraints[l], h.matrices[l]};
        const bool     success = Portable::OperatorDispatchFactory::dispatch(h.degrees[l], runner);
        AssertThrow(success, ExcMessage("No CPU operator for this degree."));
      }

    h.top_matrix = std::make_unique<CPU::LaplaceOperator<dim, top_degree, full_number>>(
      h.dof_handlers[max_level], h.top_constraints, false);

    // geometric transfers between h-levels, polynomial ones between p-levels
    h.transfers.resize(0, max_level);
    for (unsigned int l = 1; l <= max_level; ++l)
      {
        h.transfers[l] = std::make_unique<TransferType>();
        if (l < n_h_levels)
          h.transfers[l]->reinit_geometric_transfer(h.dof_handlers[l],
                                                    h.dof_handlers[l - 1],
                                                    h.constraints[l],
                                                    h.constraints[l - 1]);
        else
          h.transfers[l]->reinit_polynomial_transfer(h.dof_handlers[l],
                                                     h.dof_handlers[l - 1],
                                                     h.constraints[l],
                                                     h.constraints[l - 1]);
        h.transfers[l]->enable_inplace_operations_if_possible(
          h.matrices[l - 1]->get_vector_partitioner(), h.matrices[l]->get_vector_partitioner());
      }

    // Chebyshev smoothers; on level 0 Chebyshev serves as coarse solver
    h.smoothers.resize(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (l > 0)
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = n_smooth;
            smoother_data.eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = h.matrices[0]->m();
          }

        h.matrices[l]->compute_diagonal();
        smoother_data.preconditioner = h.matrices[l]->get_matrix_diagonal_inverse();

        h.smoothers[l].initialize(*h.matrices[l], smoother_data);
      }
  }



  // f = 1 on the top level (as poisson_cube_cpu's compute_rhs())
  template <int dim, int top_degree>
  void
  CpuSide<dim, top_degree>::compute_rhs(Hierarchy &h) const
  {
    const DoFHandler<dim>    &dof_handler = h.dof_handlers.back();
    const FiniteElement<dim> &fe          = dof_handler.get_fe();

    VectorType rhs_ghosted(dof_handler.locally_owned_dofs(),
                           DoFTools::extract_locally_relevant_dofs(dof_handler),
                           layout.comm_world);

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim>     fe_values(fe, quadrature, update_values | update_JxW_values);

    Vector<full_number>                  cell_rhs(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_rhs = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              cell_rhs(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);

          cell->get_dof_indices(dof_indices);
          h.top_constraints.distribute_local_to_global(cell_rhs, dof_indices, rhs_ghosted);
        }
    rhs_ghosted.compress(VectorOperation::add);

    h.rhs.copy_locally_owned_data_from(rhs_ghosted);
  }



  template <int dim, int top_degree>
  std::pair<unsigned int, double>
  CpuSide<dim, top_degree>::solve()
  {
    Assert(hierarchy, ExcMessage("solve() before setup()."));

    Timer time;
    const std::pair<unsigned int, double> cg = hierarchy->solver->solve_cg();

    solve_time = Utilities::MPI::max(time.wall_time(), layout.comm_world);
    if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
      std::cout << "CG on the fine level (" << hierarchy->dof_handlers.back().n_dofs()
                << " DoFs): " << cg.first << " iterations, reduction " << std::setprecision(4)
                << cg.second << " per iteration, " << solve_time << " s" << std::endl;

    return cg;
  }



  template <int dim, int top_degree>
  std::vector<LevelTimes>
  CpuSide<dim, top_degree>::benchmark_levels(const unsigned int n_calls)
  {
    Assert(hierarchy, ExcMessage("benchmark_levels() before setup()."));
    Hierarchy &h = *hierarchy;

    std::vector<LevelTimes> times(h.dof_handlers.max_level() + 1);
    for (unsigned int l = 0; l <= h.dof_handlers.max_level(); ++l)
      {
        times[l].n_dofs = h.dof_handlers[l].n_dofs();

        VectorTypeMG dst, src;
        h.matrices[l]->initialize_dof_vector(dst);
        h.matrices[l]->initialize_dof_vector(src);
        time_level_operator(layout.comm_world, *h.matrices[l], dst, src, n_calls, times[l]);

        if (l > 0)
          {
            VectorTypeMG vec_coarse;
            h.matrices[l - 1]->initialize_dof_vector(vec_coarse);
            time_level_transfer(
              layout.comm_world, *h.transfers[l], vec_coarse, dst, n_calls, times[l]);
          }
      }
    return times;
  }



  template <int dim, int top_degree>
  void
  CpuSide<dim, top_degree>::coarse_solve()
  {
    Assert(hierarchy, ExcMessage("coarse_solve() before setup()."));
    Hierarchy &h = *hierarchy;

    // GPU numbering -> CPU numbering (P), float -> double
    Timer time;
    h.permutation.gpu_to_cpu(ArrayView<const vcycle_number>(gpu_switch_values),
                             ArrayView<vcycle_number>(h.cpu_values));
    for (unsigned int i = 0; i < h.cpu_values.size(); ++i)
      h.cpu_rhs.local_element(i) = h.cpu_values[i];
    p_times[0] += time.wall_time();

    // one V-cycle on the CPU levels, zero initial guess (timed by the solver)
    h.solver->vmult(h.cpu_solution, h.cpu_rhs);

    // double -> float, CPU numbering -> GPU numbering (P back)
    time.restart();
    for (unsigned int i = 0; i < h.cpu_values.size(); ++i)
      h.cpu_values[i] = h.cpu_solution.local_element(i);
    h.permutation.cpu_to_gpu(ArrayView<vcycle_number>(h.cpu_values),
                             ArrayView<vcycle_number>(gpu_switch_values));
    p_times[1] += time.wall_time();
  }
} // namespace hybrid
