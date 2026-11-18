// Hybrid CPU-GPU multigrid: the ph-multigrid of poisson_cube_bk3 with the
// fine levels switch_level.. on the GPU ranks (comm_gpu) and the coarse
// levels 0..switch_level on all ranks (comm_world). The GPU V-cycle's coarse
// solve on the switch level is one V-cycle on the CPU levels; the CPU-only
// ranks join it through the command loop.
//
//   mpirun -n <ranks> ./program ranks_per_gpu degree [min_size] [max_size]
//                               [n_pre_smooth] [n_post_smooth] [mesh] [switch_dofs]
//                               [cpu_only_dofs]
//
// As poisson_cube_bk3 with ranks_per_gpu in front and the two thresholds at the end:
// ranks_per_gpu = k: one GPU rank + k-1 CPU-only ranks per GPU.
// degree:        polynomial degree of the fine level, 1..4
// min_size, max_size: the refinement cycles with that many DoFs are run
// n_pre_smooth, n_post_smooth: Chebyshev degree of the smoother (default 3,
//                both have to be equal)
// mesh:          d (default): doubling mesh, s: cube meshes of poisson_cube_*
// switch_dofs:   the switch level of a cycle is its finest level below the
//                fine one with at most that many DoFs per MPI rank of
//                comm_world (default 8000), level 0 if there is none
//                (switch_dofs = 0: all levels on the GPUs).
// cpu_only_dofs: a cycle whose fine level has at most that many DoFs per rank
//                runs on the CPUs only, on all ranks (default 50000, 0: never)
// Both thresholds can be comma-separated lists: the cycles are then run once
// per entry in the same program run (start-up is paid once). cpu_only_dofs
// has one entry for all or as many as switch_dofs.
// HYBRID_LAUNCH_EPOCH=$(date +%s.%N) set before the launch: the time from
// there to main() is printed next to the MPI + Kokkos initialization time.
// Mixed precision as poisson_cube_bk3: V-cycle in float (both sides), outer CG
// in double.
// The tables at the end of each cycle are those of poisson_cube_bk3, with the
// switch level and its number of DoFs as two more columns.
// Without a GPU, HYBRID_SIMULATED_RANKS_PER_NODE=<n> mimics nodes of n ranks.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hybrid_benchmark.h"
#include "hybrid_coarse_solver.h"
#include "hybrid_cpu_side.h"
#include "hybrid_gpu_side.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_service.h"
#include "hybrid_triangulations.h"


namespace hybrid
{
  // Verifies the communicators built by MPILayout. Collective on comm_world.
  void
  check_layout(const MPILayout &layout)
  {
    const unsigned int world_rank = Utilities::MPI::this_mpi_process(layout.comm_world);

    // every block has k ranks
    AssertThrow(Utilities::MPI::n_mpi_processes(layout.comm_grace) == layout.ranks_per_gpu,
                ExcMessage("Block does not have ranks_per_gpu ranks."));

    // count the GPU ranks in my block: sum of 1 (GPU rank) and 0 (CPU-only)
    AssertThrow(Utilities::MPI::sum(layout.is_gpu_rank ? 1u : 0u, layout.comm_grace) == 1u,
                ExcMessage("Block does not have exactly one GPU rank."));

    // comm_gpu has one rank per block; CPU-only ranks are not in it
    if (layout.is_gpu_rank)
      AssertThrow(Utilities::MPI::n_mpi_processes(layout.comm_gpu) == layout.n_graces,
                  ExcMessage("comm_gpu does not contain one rank per block."));
    else
      AssertThrow(layout.comm_gpu == MPI_COMM_NULL,
                  ExcMessage("CPU-only rank is a member of comm_gpu."));

    // the GPU rank (block rank 0) sends its world rank to the block: it has to
    // be the lowest one
    const unsigned int gpu_world_rank = Utilities::MPI::broadcast(layout.comm_grace, world_rank, 0);
    AssertThrow(gpu_world_rank <= world_rank, ExcMessage("GPU rank is not first in its block."));

    // every block index 0..n_graces-1 is used by exactly k ranks: each rank
    // collects all ranks' grace_index and counts
    const std::vector<unsigned int> indices =
      Utilities::MPI::all_gather(layout.comm_world, layout.grace_index);
    for (unsigned int g = 0; g < layout.n_graces; ++g)
      AssertThrow(std::count(indices.begin(), indices.end(), g) == layout.ranks_per_gpu,
                  ExcMessage("Block " + std::to_string(g) + " has the wrong number of ranks."));
  }



  // Driver on the GPU ranks: set up the GPU levels switch_level.. and the
  // CPU levels 0..switch_level (with P from the GPU side's switch level),
  // solve with CG on the fine level, then time the levels of both sides.
  template <int dim, int fe_degree, typename CpuSideType>
  CycleResult
  run_driver(const MPILayout          &layout,
             GpuSide<dim, fe_degree> &gpu_side,
             CpuSideType             &cpu_side,
             const unsigned int       n_refine,
             const unsigned int       switch_level)
  {
    // the GPU V-cycle stops at the switch level and hands it to the CPU levels
    HybridCoarseSolver<CpuSideType> coarse_solver(layout, cpu_side);
    gpu_side.setup(n_refine, switch_level, &coarse_solver);

    // P is built from the GPU side's switch level
    cpu_side.set_gpu_switch_dofs(&gpu_side.level_dofs(switch_level));
    send_command(layout, Command::setup, switch_level);
    cpu_side.setup(switch_level);

    // every V-cycle of the CG solve runs one coarse solve on all ranks;
    // timers of all parts are reset per solve, printed for the last one
    CycleResult result;
    result.time_cg = 1e10;
    for (unsigned int i = 0; i < 10; ++i)
      {
        coarse_solver.reset_times();
        cpu_side.reset_times();
        result.cg      = gpu_side.solve();
        result.time_cg = std::min(result.time_cg, gpu_side.last_solve_time());
      }
    gpu_side.print_level_times();
    cpu_side.print_times();
    coarse_solver.print_times(std::cout);

    // levels above the switch level on the GPUs, the others on the CPUs
    const unsigned int n_calls = n_calls_per_batch(gpu_side.n_fine_dofs());
    std::tie(result.best_mv, result.best_mv_smooth) = gpu_side.benchmark_matvec(n_calls);
    result.levels                                   = gpu_side.benchmark_levels(n_calls);

    send_command(layout, Command::benchmark_levels, n_calls);
    const std::vector<LevelTimes> cpu_levels = cpu_side.benchmark_levels(n_calls);
    std::copy(cpu_levels.begin(), cpu_levels.end(), result.levels.begin());

    send_command(layout, Command::stop);
    return result;
  }



  // Driver on the GPU ranks of a CPU-only cycle: all levels and the CG solve
  // on comm_world.
  template <typename CpuSideType>
  CycleResult
  run_cpu_only_driver(const MPILayout &layout, CpuSideType &cpu_side, const unsigned int fine_level)
  {
    send_command(layout, Command::setup_cpu_only, fine_level);
    cpu_side.setup(fine_level, true);

    CycleResult result;
    result.time_cg = 1e10;
    for (unsigned int i = 0; i < 10; ++i)
      {
        send_command(layout, Command::solve_cg);
        result.cg      = cpu_side.solve();
        result.time_cg = std::min(result.time_cg, cpu_side.last_solve_time());
      }
    cpu_side.print_level_times();

    const unsigned int n_calls = n_calls_per_batch(cpu_side.n_top_dofs());
    send_command(layout, Command::benchmark_matvec, n_calls);
    std::tie(result.best_mv, result.best_mv_smooth) = cpu_side.benchmark_matvec(n_calls);
    send_command(layout, Command::benchmark_levels, n_calls);
    result.levels = cpu_side.benchmark_levels(n_calls);

    send_command(layout, Command::stop);
    return result;
  }



  // cpu_degree: degree of the switch level, the top CPU level
  template <int dim, int fe_degree, int cpu_degree>
  CycleResult
  run_hybrid(const MPILayout   &layout,
             const unsigned int n_refine,
             const unsigned int switch_level,
             const CoarseGrid  &coarse_grid,
             const unsigned int n_smooth)
  {
    // GPU side only on the GPU ranks; declared first so that it outlives
    // cpu_side, which points to its switch-level DoFs
    std::unique_ptr<GpuSide<dim, fe_degree>> gpu_side;
    if (layout.is_gpu_rank)
      gpu_side = std::make_unique<GpuSide<dim, fe_degree>>(layout, coarse_grid, n_smooth);

    // CPU side on all ranks. GPU ranks drive, CPU-only ranks follow their
    // commands.
    CpuSide<dim, cpu_degree> cpu_side(layout, coarse_grid, n_smooth, n_refine, fe_degree);
    if (layout.is_gpu_rank)
      return run_driver(layout, *gpu_side, cpu_side, n_refine, switch_level);

    run_service_loop(layout, cpu_side);
    return {};
  }



  // all levels on the CPUs
  template <int dim, int fe_degree>
  CycleResult
  run_cpu_only(const MPILayout   &layout,
               const unsigned int n_refine,
               const unsigned int fine_level,
               const CoarseGrid  &coarse_grid,
               const unsigned int n_smooth)
  {
    CpuSide<dim, fe_degree> cpu_side(layout, coarse_grid, n_smooth, n_refine, fe_degree);
    if (layout.is_gpu_rank)
      return run_cpu_only_driver(layout, cpu_side, fine_level);

    run_service_loop(layout, cpu_side);
    return {};
  }



  // One cycle. Dispatches the degree of the switch level (known at run
  // time); switch_level = fine level: CPU-only cycle.
  template <int dim, int fe_degree>
  CycleResult
  run(const MPILayout   &layout,
      const unsigned int n_refine,
      const unsigned int switch_level,
      const CoarseGrid  &coarse_grid,
      const unsigned int n_smooth)
  {
    const std::vector<unsigned int> degrees = ph_level_degrees(n_refine + 1, fe_degree);
    AssertThrow(switch_level < degrees.size(),
                ExcMessage("The switch level is above the fine level."));

    if (switch_level + 1 == degrees.size())
      return run_cpu_only<dim, fe_degree>(layout, n_refine, switch_level, coarse_grid, n_smooth);

    // below Q4 the p-levels are 4 -> 2 -> 1, 3 -> 1, 2 -> 1
    if (degrees[switch_level] == 1)
      return run_hybrid<dim, fe_degree, 1>(layout, n_refine, switch_level, coarse_grid, n_smooth);

    if constexpr (fe_degree > 2)
      {
        AssertThrow(degrees[switch_level] == 2, ExcNotImplemented());
        return run_hybrid<dim, fe_degree, 2>(
          layout, n_refine, switch_level, coarse_grid, n_smooth);
      }

    AssertThrow(false, ExcInternalError());
    return {};
  }



  // The refinement cycles of poisson_cube_bk3 with min_size..max_size DoFs.
  template <int dim, int fe_degree>
  void
  run_cycles(const MPILayout              &layout,
             const std::size_t             min_size,
             const std::size_t             max_size,
             const unsigned int            n_smooth,
             const bool                    use_doubling_mesh,
             const types::global_dof_index switch_dofs,
             const types::global_dof_index cpu_only_dofs)
  {
    const bool         is_root = Utilities::MPI::this_mpi_process(layout.comm_world) == 0;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(layout.comm_world);

    // per-rank threshold -> global number of DoFs, without overflow
    const auto global_limit = [n_ranks](const types::global_dof_index per_rank) {
      const types::global_dof_index max = std::numeric_limits<types::global_dof_index>::max();
      return per_rank > max / n_ranks ? max : per_rank * n_ranks;
    };

    if (is_root)
      std::cout << "Testing FE_Q<" << dim << ">(" << fe_degree << ")" << std::endl;

    const unsigned int sizes[] = {1,   2,   3,   4,   5,   6,   7,   8,   10,  12,   14,   16,  20,
                                  24,  28,  32,  40,  48,  56,  64,  80,  96,  112,  128,  160, 192,
                                  224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536};

    CycleTables tables;

    for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int); ++cycle)
      {
        if (is_root)
          std::cout << "Cycle " << cycle << std::endl;

        // doubling mesh: cycle = 3 * n_refine + stretch; else the coarse grid
        // has the odd factor of sizes[cycle] cells per direction
        unsigned int n_refine = 0;
        CoarseGrid   coarse_grid;
        if (use_doubling_mesh)
          {
            n_refine    = cycle / 3;
            coarse_grid = CoarseGrid{1, static_cast<int>(cycle % 3)};
          }
        else
          {
            unsigned int n_subdivisions = sizes[cycle];
            while (n_subdivisions > 1 && n_subdivisions % 2 == 0)
              {
                ++n_refine;
                n_subdivisions /= 2;
              }
            coarse_grid = CoarseGrid{n_subdivisions, -1};
          }

        const std::vector<types::global_dof_index> n_dofs =
          ph_level_n_dofs<dim>(coarse_grid, n_refine, fe_degree);

        if (n_dofs.back() < min_size)
          continue;

        if (n_dofs.back() > max_size)
          {
            if (is_root)
              std::cout << "Projected size " << n_dofs.back()
                        << " higher than max size, terminating." << std::endl
                        << std::endl;
            break;
          }

        // CPU-only if the fine level is small enough; else switch below it
        const unsigned int fine_level = n_dofs.size() - 1;
        const bool cpu_only = fine_level == 0 || n_dofs.back() <= global_limit(cpu_only_dofs);
        const unsigned int switch_level =
          cpu_only ? fine_level :
                     std::min(find_switch_level(n_dofs, global_limit(switch_dofs)), fine_level - 1);

        if (is_root)
          {
            std::cout << "Number of degrees of freedom: " << n_dofs.back() << std::endl
                      << "Switch level " << switch_level << " of 0.." << fine_level << " with "
                      << n_dofs[switch_level] << " DoFs (" << n_dofs[switch_level] / n_ranks
                      << " per rank): ";
            if (switch_level == fine_level)
              std::cout << "CPU-only cycle" << std::endl;
            else
              std::cout << "levels 0.." << switch_level << " on the CPUs, " << switch_level + 1
                        << ".." << fine_level << " on the GPUs" << std::endl;
          }

        const CycleResult result =
          run<dim, fe_degree>(layout, n_refine, switch_level, coarse_grid, n_smooth);

        if (is_root)
          {
            types::global_cell_index n_cells = 1;
            for (unsigned int d = 0; d < dim; ++d)
              n_cells *= types::global_cell_index(n_coarse_cells(coarse_grid, d)) << n_refine;

            std::cout << std::endl;
            tables.add_cycle(n_ranks,
                             n_cells,
                             switch_level,
                             result,
                             std::cout);
            tables.print(std::cout);
          }
      }
  }
} // namespace hybrid



int
main(int argc, char **argv)
{
  using namespace dealii;

  // seconds since the epoch, comparable with `date +%s.%N` of the job script
  const auto epoch_now = []() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
      .count();
  };
  const double time_main = epoch_now();

  // starts MPI and Kokkos, 1 thread per rank; shuts both down at the end of
  // main, after `layout` below has freed its communicators
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const double time_ready = epoch_now();

  try
    {
      const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

      // all ranks return 1; only rank 0 prints
      if (argc < 3)
        {
          if (is_root)
            std::cout << "Usage: mpirun -n <ranks> ./program ranks_per_gpu degree [min_size] "
                         "[max_size] [n_pre_smooth] [n_post_smooth] [mesh] [switch_dofs] "
                         "[cpu_only_dofs]"
                      << std::endl;
          return 1;
        }

      const unsigned int ranks_per_gpu = std::atoi(argv[1]);
      const unsigned int degree        = std::atoi(argv[2]);
      const std::size_t  min_size      = argc > 3 ? std::atoll(argv[3]) : 1;
      const std::size_t  max_size =
        argc > 4 ? std::atoll(argv[4]) : std::numeric_limits<std::size_t>::max();
      const unsigned int n_pre_smooth      = argc > 5 ? std::atoi(argv[5]) : 3;
      const unsigned int n_post_smooth     = argc > 6 ? std::atoi(argv[6]) : n_pre_smooth;
      const bool         use_doubling_mesh = argc > 7 ? argv[7][0] == 'd' : true;
      // comma-separated list of thresholds
      const auto parse_list = [](const std::string &text) {
        std::vector<types::global_dof_index> values;
        std::stringstream                    stream(text);
        for (std::string entry; std::getline(stream, entry, ',');)
          values.push_back(std::atoll(entry.c_str()));
        return values;
      };
      const std::vector<types::global_dof_index> switch_dofs =
        parse_list(argc > 8 ? argv[8] : "8000");
      std::vector<types::global_dof_index> cpu_only_dofs =
        parse_list(argc > 9 ? argv[9] : "50000");
      AssertThrow(degree >= 1 && degree <= 4, ExcMessage("degree has to be 1..4."));
      AssertThrow(n_pre_smooth == n_post_smooth,
                  ExcMessage("n_pre_smooth and n_post_smooth have to be equal."));
      AssertThrow(!switch_dofs.empty() && !cpu_only_dofs.empty(),
                  ExcMessage("Empty threshold list."));
      if (cpu_only_dofs.size() == 1)
        cpu_only_dofs.resize(switch_dofs.size(), cpu_only_dofs[0]);
      AssertThrow(cpu_only_dofs.size() == switch_dofs.size(),
                  ExcMessage("cpu_only_dofs needs one entry or as many as switch_dofs."));

      // start-up times, slowest rank
      {
        const char  *launch      = std::getenv("HYBRID_LAUNCH_EPOCH");
        const double launch_time = Utilities::MPI::max(
          launch != nullptr ? time_main - std::atof(launch) : 0., MPI_COMM_WORLD);
        const double init_time = Utilities::MPI::max(time_ready - time_main, MPI_COMM_WORLD);
        if (is_root)
          {
            std::cout << "Start-up, slowest rank: ";
            if (launch != nullptr)
              std::cout << "launch to main() " << launch_time << " s, ";
            std::cout << "MPI + Kokkos initialization " << init_time << " s" << std::endl
                      << std::endl;
          }
      }

      // build the communicators, print the table, verify
      hybrid::MPILayout layout(MPI_COMM_WORLD, ranks_per_gpu);
      layout.print(std::cout);
      hybrid::check_layout(layout);

      // only warnings: on pc02 they never fire; on JUPITER they catch a wrong
      // srun rank placement or --ntasks-per-node that does not match k
      const bool share_socket = layout.blocks_share_socket();
      const bool match_gpus   = layout.blocks_match_slurm_gpus();
      if (is_root)
        {
          if (!share_socket)
            std::cout << "WARNING: the ranks of some block are not pinned to one CPU socket."
                      << std::endl
                      << "         With srun use --distribution=block:block --cpu-bind=cores."
                      << std::endl;
          if (!match_gpus)
            std::cout << "WARNING: ranks per node / ranks_per_gpu differs from SLURM_GPUS_ON_NODE."
                      << std::endl;
          std::cout << "MPI layout checks passed." << std::endl << std::endl;
        }

      // one pass over the cycles per pair of thresholds; degree known at run
      // time, templates need it at compile time
      constexpr int dim = 3;
      for (unsigned int i = 0; i < switch_dofs.size(); ++i)
        {
          if (is_root)
            std::cout << "Settings of parameters:                " << std::endl
                      << "Number of MPI ranks:                   "
                      << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                      << "Ranks per GPU:                         " << ranks_per_gpu << std::endl
                      << "V-cycle number type:                   "
                      << (std::is_same_v<hybrid::GpuSide<3, 1>::vcycle_number, float> ? "float" :
                                                                                         "double")
                      << std::endl
                      << "Polynomial degree:                     " << degree << std::endl
                      << "Minimum size:                          " << min_size << std::endl
                      << "Maximum size:                          " << max_size << std::endl
                      << "Number of pre-smoother iters:          " << n_pre_smooth << std::endl
                      << "Number of post-smoother iters:         " << n_post_smooth << std::endl
                      << "Use doubling mesh:                     " << use_doubling_mesh << std::endl
                      << "Switch level, DoFs per rank:           " << switch_dofs[i] << std::endl
                      << "CPU-only cycle, DoFs per rank:         " << cpu_only_dofs[i] << std::endl
                      << "Wait of the CPU-only ranks:            "
                      << (hybrid::idle_sleep_microseconds() == 0 ?
                            std::string("blocking (busy)") :
                            "sleep " + std::to_string(hybrid::idle_sleep_microseconds()) + " us")
                      << std::endl
                      << std::endl;

          switch (degree)
            {
              case 1:
                hybrid::run_cycles<dim, 1>(layout,
                                           min_size,
                                           max_size,
                                           n_pre_smooth,
                                           use_doubling_mesh,
                                           switch_dofs[i],
                                           cpu_only_dofs[i]);
                break;
              case 2:
                hybrid::run_cycles<dim, 2>(layout,
                                           min_size,
                                           max_size,
                                           n_pre_smooth,
                                           use_doubling_mesh,
                                           switch_dofs[i],
                                           cpu_only_dofs[i]);
                break;
              case 3:
                hybrid::run_cycles<dim, 3>(layout,
                                           min_size,
                                           max_size,
                                           n_pre_smooth,
                                           use_doubling_mesh,
                                           switch_dofs[i],
                                           cpu_only_dofs[i]);
                break;
              case 4:
                hybrid::run_cycles<dim, 4>(layout,
                                           min_size,
                                           max_size,
                                           n_pre_smooth,
                                           use_doubling_mesh,
                                           switch_dofs[i],
                                           cpu_only_dofs[i]);
                break;
            }
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
