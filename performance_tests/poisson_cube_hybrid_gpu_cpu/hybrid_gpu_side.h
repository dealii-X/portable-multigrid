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
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "hybrid_benchmark.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_triangulations.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_laplace_operator_bk3.h"
#include "portable_multigrid_solver.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * GPU levels of the hybrid multigrid on comm_gpu: the ph-hierarchy of
   * poisson_cube_bk3 (BK3 operators, Portable transfers, Chebyshev), copied
   * from there. Levels 0..n_refine are Q1 on the meshes refined l times, the
   * levels above are p-levels on the fine mesh up to Q_fe_degree.
   *
   * setup() can cut the hierarchy at a lowest level min_level (the switch
   * level s-1 of the hybrid, h- or p-level) and use another coarse solver
   * there.
   */
  template <int dim, int fe_degree>
  class GpuSide
  {
  public:
    using vcycle_number = float; // as in poisson_cube_bk3
    using full_number   = double;

    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>;
    using VectorType   = LinearAlgebra::distributed::Vector<full_number, MemorySpace::Default>;

    using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorTypeMG>;
    using FineMatrixType  = Portable::LaplaceOperatorBase<dim, VectorType>;
    using TransferType    = Portable::MGTransferBase<dim, vcycle_number>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;

    using MGSolverType =
      multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>;

    GpuSide(const MPILayout &layout, const CoarseGrid &coarse_grid, const unsigned int n_smooth)
      : layout(layout)
      , coarse_grid(coarse_grid)
      , n_smooth(n_smooth)
    {
      Assert(layout.is_gpu_rank, ExcMessage("GpuSide lives on the GPU ranks only."));
    }

    // Collective on comm_gpu: build the levels min_level.. for a fine mesh
    // refined n_refine times (any level below the fine one). On min_level the
    // V-cycle calls coarse_solver, or if nullptr Chebyshev as coarse solver
    // (as on level 0 of poisson_cube_bk3).
    void
    setup(const unsigned int                  n_refine,
          const unsigned int                  min_level     = 0,
          const MGCoarseGridBase<VectorTypeMG> *coarse_solver = nullptr);

    // Collective on comm_gpu: CG on the fine level with f = 1, preconditioned
    // by the V-cycle. Returns iterations and reduction per iteration.
    std::pair<unsigned int, double>
    solve();

    // Wall time of the last solve(), maximum over the ranks.
    double
    last_solve_time() const
    {
      return solve_time;
    }

    types::global_dof_index
    n_fine_dofs() const
    {
      return hierarchy->dof_handlers.back().n_dofs();
    }

    // Collective on comm_gpu: best times of the fine-level matvec in double
    // and in the V-cycle number type.
    std::pair<double, double>
    benchmark_matvec(const unsigned int n_calls)
    {
      return time_fine_matvecs(layout.comm_gpu, *hierarchy->solver, n_fine_dofs(), n_calls);
    }

    // Collective on comm_gpu: best times of the operators and transfers of
    // the GPU levels above the lowest one (entry l = level l; the levels up
    // to the lowest one are left empty).
    std::vector<LevelTimes>
    benchmark_levels(const unsigned int n_calls);

    // Rank 0: per-level times of the last solve() (MultigridSolver's timers).
    void
    print_level_times()
    {
      if (Utilities::MPI::this_mpi_process(layout.comm_gpu) == 0)
        std::cout << "==== GPU side levels, rank 0, last solve" << std::endl;
      hierarchy->solver->print_wall_times();
    }

    // DoFHandler of a GPU level (index in the full ph-hierarchy).
    const DoFHandler<dim> &
    level_dofs(const unsigned int level) const
    {
      Assert(level >= hierarchy->dof_handlers.min_level() &&
               level <= hierarchy->dof_handlers.max_level(),
             ExcMessage("This level is not on the GPU."));
      return hierarchy->dof_handlers[level];
    }

  private:
    // Everything one setup() builds. Members are destroyed in reverse order:
    // solver first, mesh last.
    struct Hierarchy
    {
      Hierarchy(const MPI_Comm comm, const CoarseGrid &coarse_grid, const unsigned int n_refine)
        : mesh(comm, coarse_grid, n_refine)
      {}

      MeshLevels<dim>                                 mesh;
      std::vector<unsigned int>                       degrees; // polynomial degree per level
      MGLevelObject<DoFHandler<dim>>                  dof_handlers;
      MGLevelObject<AffineConstraints<vcycle_number>> constraints;
      AffineConstraints<full_number>                  fine_constraints; // fine level, double
      MGLevelObject<std::unique_ptr<LevelMatrixType>> matrices;
      std::unique_ptr<FineMatrixType>                 fine_matrix; // fine level, double
      MGLevelObject<std::unique_ptr<TransferType>>    transfers;
      MGLevelObject<SmootherType>                     smoothers;
      VectorType                                      rhs;
      std::unique_ptr<MGSolverType>                   solver;
    };

    // creates the BK3 operator of the dispatched degree in `matrix`
    struct OperatorRunner
    {
      const DoFHandler<dim>                  &dof_handler;
      const AffineConstraints<vcycle_number> &constraints;
      std::unique_ptr<LevelMatrixType>       &matrix;

      template <unsigned int degree>
      void
      run()
      {
        matrix = std::make_unique<Portable::LaplaceOperatorBK3<dim, degree, vcycle_number>>(
          dof_handler, constraints, false);
      }
    };

    // creates the p-transfer of the dispatched degree pair in `transfer`
    struct PolynomialTransferRunner
    {
      const Portable::OperatorInfoBase<dim>          &op_coarse;
      const Portable::OperatorInfoBase<dim>          &op_fine;
      const AffineConstraints<vcycle_number>         &constraints_coarse;
      const AffineConstraints<vcycle_number>         &constraints_fine;
      std::unique_ptr<TransferType>                  &transfer;

      template <unsigned int degree_coarse, unsigned int degree_fine>
      void
      run()
      {
        transfer = std::make_unique<
          Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, vcycle_number>>();
        transfer->reinit(op_coarse, op_fine, constraints_coarse, constraints_fine);
      }
    };

    void
    setup_dofs(Hierarchy &h, const unsigned int min_level) const;

    // with_coarse_smoother: Chebyshev as coarse solver on the lowest level
    void
    setup_operators(Hierarchy &h, const bool with_coarse_smoother) const;

    void
    compute_rhs(Hierarchy &h) const;

    const MPILayout   &layout;
    const CoarseGrid coarse_grid;
    const unsigned int n_smooth;

    double solve_time = 0.;

    std::unique_ptr<Hierarchy> hierarchy;
  };



  template <int dim, int fe_degree>
  void
  GpuSide<dim, fe_degree>::setup(const unsigned int                    n_refine,
                                 const unsigned int                    min_level,
                                 const MGCoarseGridBase<VectorTypeMG> *coarse_solver)
  {
    Kokkos::fence();
    Timer time;

    hierarchy.reset();
    hierarchy = std::make_unique<Hierarchy>(layout.comm_gpu, coarse_grid, n_refine);
    Hierarchy &h = *hierarchy;

    print_level_partitions(h.mesh.levels, layout.comm_gpu, "GPU side, comm_gpu", std::cout);

    setup_dofs(h, min_level);
    setup_operators(h, coarse_solver == nullptr);
    compute_rhs(h);

    h.solver = std::make_unique<MGSolverType>(h.fine_matrix,
                                              h.dof_handlers,
                                              h.constraints,
                                              h.transfers,
                                              h.matrices,
                                              h.smoothers,
                                              h.rhs,
                                              n_smooth,
                                              n_smooth,
                                              coarse_solver);
    Kokkos::fence();

    const double setup_time = Utilities::MPI::max(time.wall_time(), layout.comm_gpu);
    if (Utilities::MPI::this_mpi_process(layout.comm_gpu) == 0)
      {
        std::cout << "GPU levels " << h.dof_handlers.min_level() << ".."
                  << h.dof_handlers.max_level() << ", degree/DoFs:";
        for (unsigned int l = h.dof_handlers.min_level(); l <= h.dof_handlers.max_level(); ++l)
          std::cout << " Q" << h.degrees[l] << '/' << h.dof_handlers[l].n_dofs();
        std::cout << std::endl << "GPU side setup: " << setup_time << " s" << std::endl;
      }
  }



  template <int dim, int fe_degree>
  void
  GpuSide<dim, fe_degree>::setup_dofs(Hierarchy &h, const unsigned int min_level) const
  {
    // h-levels: Q1 on the meshes refined 0..n_refine times; then p-levels on
    // the fine mesh with degrees ..., fe_degree - 2, fe_degree (as bk3)
    const unsigned int n_h_levels = h.mesh.levels.size();
    h.degrees                     = ph_level_degrees(n_h_levels, fe_degree);

    const unsigned int max_level = h.degrees.size() - 1;
    AssertThrow(min_level <= max_level, ExcMessage("min_level is above the fine level."));

    h.dof_handlers.resize(min_level, max_level);
    h.constraints.resize(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        DoFHandler<dim> &dof_handler = h.dof_handlers[l];
        dof_handler.reinit(*h.mesh.levels[std::min(l, n_h_levels - 1)]);
        dof_handler.distribute_dofs(FE_Q<dim>(h.degrees[l]));

        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        // built in double, as in poisson_cube_bk3; kept for the fine level
        AffineConstraints<full_number>  level_constraints_double;
        AffineConstraints<full_number> &constraints =
          l == max_level ? h.fine_constraints : level_constraints_double;

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



  template <int dim, int fe_degree>
  void
  GpuSide<dim, fe_degree>::setup_operators(Hierarchy &h, const bool with_coarse_smoother) const
  {
    const unsigned int min_level  = h.dof_handlers.min_level();
    const unsigned int max_level  = h.dof_handlers.max_level();
    const unsigned int n_h_levels = h.mesh.levels.size();

    // BK3 level operators in vcycle_number, the fine level also in double
    h.matrices.resize(min_level, max_level);
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        OperatorRunner runner{h.dof_handlers[l], h.constraints[l], h.matrices[l]};
        const bool     success = Portable::OperatorDispatchFactory::dispatch(h.degrees[l], runner);
        AssertThrow(success, ExcMessage("No BK3 operator for this degree."));
      }

    h.fine_matrix = std::make_unique<Portable::LaplaceOperatorBK3<dim, fe_degree, full_number>>(
      h.dof_handlers[max_level], h.fine_constraints, false);

    // geometric transfers between h-levels, polynomial ones between p-levels
    h.transfers.resize(min_level, max_level);
    for (unsigned int l = min_level + 1; l <= max_level; ++l)
      if (l < n_h_levels)
        {
          h.transfers[l] = std::make_unique<Portable::GeometricTransfer<dim, 1, vcycle_number>>();
          h.transfers[l]->reinit(*h.matrices[l - 1],
                                 *h.matrices[l],
                                 h.constraints[l - 1],
                                 h.constraints[l]);
        }
      else
        {
          PolynomialTransferRunner runner{*h.matrices[l - 1],
                                          *h.matrices[l],
                                          h.constraints[l - 1],
                                          h.constraints[l],
                                          h.transfers[l]};
          const bool               success =
            Portable::PolynomialTransferDispatchFactory::dispatch(h.degrees[l - 1],
                                                                  h.degrees[l],
                                                                  runner);
          AssertThrow(success, ExcMessage("No polynomial transfer for this degree pair."));
        }

    // Chebyshev smoothers; on the lowest level either Chebyshev as coarse
    // solver or nothing (another coarse solver is used there)
    h.smoothers.resize(min_level, max_level);
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (l > min_level)
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = n_smooth;
            smoother_data.eig_cg_n_iterations = 10;
          }
        else if (with_coarse_smoother)
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = h.matrices[l]->m();
          }
        else
          continue;

        h.matrices[l]->compute_diagonal();
        smoother_data.preconditioner = h.matrices[l]->get_matrix_diagonal_inverse();
        smoother_data.constraints.copy_from(h.constraints[l]);

        h.smoothers[l].initialize(*h.matrices[l], smoother_data);
      }
  }



  // f = 1 on the fine level, assembled on the host and copied to the device
  // (as poisson_cube_bk3's compute_rhs()).
  template <int dim, int fe_degree>
  void
  GpuSide<dim, fe_degree>::compute_rhs(Hierarchy &h) const
  {
    const DoFHandler<dim>    &dof_handler = h.dof_handlers[h.dof_handlers.max_level()];
    const FiniteElement<dim> &fe          = dof_handler.get_fe();

    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host> rhs_host(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      layout.comm_gpu);

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
          h.fine_constraints.distribute_local_to_global(cell_rhs, dof_indices, rhs_host);
        }
    rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<full_number> rw_vector(dof_handler.locally_owned_dofs());
    rw_vector.import_elements(rhs_host, VectorOperation::insert);

    h.fine_matrix->initialize_dof_vector(h.rhs);
    h.rhs.import_elements(rw_vector, VectorOperation::insert);
  }



  template <int dim, int fe_degree>
  std::pair<unsigned int, double>
  GpuSide<dim, fe_degree>::solve()
  {
    Assert(hierarchy, ExcMessage("solve() before setup()."));

    Kokkos::fence();
    Timer time;
    const std::pair<unsigned int, double> cg = hierarchy->solver->solve_cg();
    Kokkos::fence();

    solve_time = Utilities::MPI::max(time.wall_time(), layout.comm_gpu);
    if (Utilities::MPI::this_mpi_process(layout.comm_gpu) == 0)
      std::cout << "CG on the fine level (" << hierarchy->dof_handlers.back().n_dofs()
                << " DoFs): " << cg.first << " iterations, reduction " << std::setprecision(4)
                << cg.second << " per iteration, " << solve_time << " s" << std::endl;

    return cg;
  }



  template <int dim, int fe_degree>
  std::vector<LevelTimes>
  GpuSide<dim, fe_degree>::benchmark_levels(const unsigned int n_calls)
  {
    Assert(hierarchy, ExcMessage("benchmark_levels() before setup()."));
    Hierarchy &h = *hierarchy;

    std::vector<LevelTimes> times(h.dof_handlers.max_level() + 1);
    for (unsigned int l = h.dof_handlers.min_level() + 1; l <= h.dof_handlers.max_level(); ++l)
      {
        times[l].n_dofs = h.dof_handlers[l].n_dofs();
        times[l].on_gpu = true;

        VectorTypeMG dst, src, vec_coarse;
        h.matrices[l]->initialize_dof_vector(dst);
        h.matrices[l]->initialize_dof_vector(src);
        h.matrices[l - 1]->initialize_dof_vector(vec_coarse);

        time_level_operator(layout.comm_gpu, *h.matrices[l], dst, src, n_calls, times[l]);
        time_level_transfer(layout.comm_gpu, *h.transfers[l], vec_coarse, dst, n_calls, times[l]);
      }
    return times;
  }
} // namespace hybrid
