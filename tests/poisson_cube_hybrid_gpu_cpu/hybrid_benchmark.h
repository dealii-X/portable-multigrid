#pragma once

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace hybrid
{
  using namespace dealii;

  // Best per-call times of one level, measured as in poisson_cube_bk3 and
  // poisson_cube_cpu: best of 5 batches, per batch the slowest rank.
  struct LevelTimes
  {
    types::global_dof_index n_dofs = 0;
    bool                    on_gpu = false;

    // transfer between this level and the one below
    double restriction  = 0.;
    double prolongation = 0.;

    // vmult_dummy of the level operator
    double mv_both         = 0.;
    double mv_ghost_only   = 0.;
    double mv_compute_only = 0.;
  };



  // What one refinement cycle reports. Filled on the GPU ranks.
  struct CycleResult
  {
    std::vector<LevelTimes>         levels; // 0..fine level
    double                          best_mv        = 0.;
    double                          best_mv_smooth = 0.;
    double                          time_cg        = 0.;
    std::pair<unsigned int, double> cg             = {0, 0.};
  };



  // calls per batch, as in poisson_cube_bk3
  inline unsigned int
  n_calls_per_batch(const types::global_dof_index n_dofs_fine)
  {
    return n_dofs_fine < 10000000 ? 200 : 50;
  }



  // One batch: time per call of f, min/max/avg over the ranks of comm.
  template <typename F>
  Utilities::MPI::MinMaxAvg
  time_per_call(const MPI_Comm comm, const unsigned int n_calls, const F &f)
  {
    Kokkos::fence();
    Timer time;
    for (unsigned int i = 0; i < n_calls; ++i)
      f();
    Kokkos::fence();

    return Utilities::MPI::min_max_avg(time.wall_time() / n_calls, comm);
  }



  // vmult_dummy with ghost exchange and computation, exchange only,
  // computation only. dst, src: vectors of the level.
  template <typename MatrixType, typename VectorType>
  void
  time_level_operator(const MPI_Comm     comm,
                      const MatrixType  &matrix,
                      VectorType        &dst,
                      VectorType        &src,
                      const unsigned int n_calls,
                      LevelTimes        &times)
  {
    times.mv_both = times.mv_ghost_only = times.mv_compute_only = 1e10;

    for (unsigned int i = 0; i < 5; ++i)
      {
        times.mv_both = std::min(times.mv_both, time_per_call(comm, n_calls, [&]() {
                                   matrix.vmult_dummy(dst, src, true, true);
                                 }).max);
        times.mv_ghost_only =
          std::min(times.mv_ghost_only, time_per_call(comm, n_calls, [&]() {
                     matrix.vmult_dummy(dst, src, true, false);
                   }).max);
        times.mv_compute_only =
          std::min(times.mv_compute_only, time_per_call(comm, n_calls, [&]() {
                     matrix.vmult_dummy(dst, src, false, true);
                   }).max);
      }
  }



  // prolongate_and_add and restrict_and_add between a level and the one below
  template <typename TransferType, typename VectorType>
  void
  time_level_transfer(const MPI_Comm      comm,
                      const TransferType &transfer,
                      VectorType         &vec_coarse,
                      VectorType         &vec_fine,
                      const unsigned int  n_calls,
                      LevelTimes         &times)
  {
    times.prolongation = times.restriction = 1e10;

    for (unsigned int i = 0; i < 5; ++i)
      times.prolongation =
        std::min(times.prolongation, time_per_call(comm, n_calls, [&]() {
                   transfer.prolongate_and_add(vec_fine, vec_coarse);
                 }).max);

    for (unsigned int i = 0; i < 5; ++i)
      times.restriction = std::min(times.restriction, time_per_call(comm, n_calls, [&]() {
                                     transfer.restrict_and_add(vec_coarse, vec_fine);
                                   }).max);
  }



  // Matvec of the fine level in double (outer CG) and in the V-cycle number
  // type. Returns the two best times; rank 0 of comm prints the double ones.
  template <typename SolverType>
  std::pair<double, double>
  time_fine_matvecs(const MPI_Comm                comm,
                    SolverType                   &solver,
                    const types::global_dof_index n_dofs,
                    const unsigned int            n_calls)
  {
    double best_mv = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const Utilities::MPI::MinMaxAvg stat =
          time_per_call(comm, n_calls, [&]() { solver.do_matvec(); });
        best_mv = std::min(best_mv, stat.max);

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                    << " " << stat.max << " [p" << stat.max_index << "]"
                    << " DoFs/s: " << n_dofs / stat.max << std::endl;
      }

    double best_mv_smooth = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      best_mv_smooth = std::min(best_mv_smooth, time_per_call(comm, n_calls, [&]() {
                                  solver.do_matvec_smoother();
                                }).max);

    return {best_mv, best_mv_smooth};
  }



  // The per-cycle output of poisson_cube_bk3 / poisson_cube_cpu: "Best
  // timings" lines and the two tables, which get one row per cycle and are
  // printed again after each. Two more columns: the switch level and its
  // number of DoFs (the fine level in a CPU-only cycle).
  class CycleTables
  {
  public:
    void
    add_cycle(const unsigned int             n_proc,
              const types::global_cell_index n_cells,
              const unsigned int             switch_level,
              const CycleResult             &result,
              std::ostream                  &out)
    {
      const LevelTimes &fine = result.levels.back();

      out << "Best timings for ndof = " << fine.n_dofs << "   mv " << result.best_mv
          << "    mv smooth " << result.best_mv_smooth << "   cg-mg " << result.time_cg
          << std::endl;

      for (unsigned int l = 1; l < result.levels.size(); ++l)
        out << "Best timings for ndof = " << result.levels[l].n_dofs << "   on level " << l
            << "|  restriction = " << result.levels[l].restriction
            << "   prolongation  =  " << result.levels[l].prolongation << side(result.levels[l])
            << std::endl;
      out << std::endl << std::endl << std::endl;

      for (unsigned int l = 0; l < result.levels.size(); ++l)
        out << "Best timings for ndof = " << result.levels[l].n_dofs << "   on level " << l
            << "|  ghost & compute =  " << result.levels[l].mv_both
            << "   ghost only      =  " << result.levels[l].mv_ghost_only
            << "   compute only    =  " << result.levels[l].mv_compute_only
            << side(result.levels[l]) << std::endl;
      out << std::endl << std::endl;

      convergence_table.add_value("n_proc", n_proc);
      convergence_table.add_value("cells", n_cells);
      convergence_table.add_value("dofs", fine.n_dofs);
      convergence_table.add_value("mv_outer", result.best_mv);
      convergence_table.add_value("mv_inner", result.best_mv_smooth);
      convergence_table.add_value("cg_time", result.time_cg);
      convergence_table.add_value("cg_its", result.cg.first);
      convergence_table.add_value("cg_reduction", result.cg.second);
      convergence_table.add_value("switch_level", switch_level);
      convergence_table.add_value("switch_dofs", result.levels[switch_level].n_dofs);

      ghost_timing_table.add_value("cells", n_cells);
      ghost_timing_table.add_value("dofs", fine.n_dofs);
      ghost_timing_table.add_value("mv_ghost_and_compute", fine.mv_both);
      ghost_timing_table.add_value("mv_compute_only", fine.mv_compute_only);
      ghost_timing_table.add_value("mv_ghost_only", fine.mv_ghost_only);
    }

    void
    print(std::ostream &out)
    {
      for (const char *column : {"mv_outer", "mv_inner", "cg_reduction", "cg_time"})
        {
          convergence_table.set_scientific(column, true);
          convergence_table.set_precision(column, 3);
        }
      convergence_table.write_text(out);
      out << std::endl << std::endl;

      for (const char *column : {"mv_ghost_and_compute", "mv_compute_only", "mv_ghost_only"})
        {
          ghost_timing_table.set_scientific(column, true);
          ghost_timing_table.set_precision(column, 4);
        }
      ghost_timing_table.write_text(out);
      out << std::endl << std::endl;
    }

  private:
    static const char *
    side(const LevelTimes &level)
    {
      return level.on_gpu ? "   (GPU)" : "   (CPU)";
    }

    ConvergenceTable convergence_table;
    ConvergenceTable ghost_timing_table;
  };
} // namespace hybrid
