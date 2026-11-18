#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <chrono>
#include <cstdlib>
#include <thread>
#include <utility>

#ifdef __linux__
#  include <sys/prctl.h>
#endif

#include "hybrid_mpi_layout.h"

namespace hybrid
{
  using namespace dealii;

  // What the GPU rank of a block tells its CPU-only ranks to do next.
  enum class Command : int
  {
    setup = 0, // build the CPU levels; payload = switch level s
    solve = 1, // one coarse solve on the CPU levels
    stop  = 2, // leave the service loop

    setup_cpu_only   = 3, // build all levels and the right-hand side; payload = fine level
    solve_cg         = 4, // CG on the fine level (CPU-only cycle)
    benchmark_levels = 5, // time the level operators and transfers; payload = calls per batch
    benchmark_matvec = 6  // time the fine-level matvecs; payload = calls per batch
  };



  // How a CPU-only rank waits for its next command: microseconds of sleep
  // between two checks, default 50, set with HYBRID_IDLE_SLEEP_US (same value
  // on all ranks). 0: blocking MPI_Bcast, which polls and keeps the core busy;
  // busy cores slow the GPU of the same GH200 at full load (jobs 2885286,
  // 2886001).
  inline unsigned int
  idle_sleep_microseconds()
  {
    static const unsigned int value = []() {
      const char *env = std::getenv("HYBRID_IDLE_SLEEP_US");
      return env != nullptr ? static_cast<unsigned int>(std::atoi(env)) : 50u;
    }();
    return value;
  }

  const int command_tag = 4711;



  // GPU rank: send the next command to its block; matches receive_command()
  // on the CPU-only ranks of the block.
  inline void
  send_command(const MPILayout &layout, const Command command, const unsigned int payload = 0)
  {
    Assert(layout.is_gpu_rank, ExcMessage("Only GPU ranks send commands."));

    int message[2] = {static_cast<int>(command), static_cast<int>(payload)};

    if (idle_sleep_microseconds() == 0)
      {
        const int ierr = MPI_Bcast(message, 2, MPI_INT, 0, layout.comm_grace);
        AssertThrowMPI(ierr);
        return;
      }

    // one message per rank: sleeping ranks would delay the tree of a broadcast
    const int n_ranks = Utilities::MPI::n_mpi_processes(layout.comm_grace);
    for (int rank = 1; rank < n_ranks; ++rank)
      {
        const int ierr = MPI_Send(message, 2, MPI_INT, rank, command_tag, layout.comm_grace);
        AssertThrowMPI(ierr);
      }
  }



  // CPU-only rank: wait until the GPU rank of the block sends a command.
  inline std::pair<Command, unsigned int>
  receive_command(const MPILayout &layout)
  {
    Assert(!layout.is_gpu_rank, ExcMessage("GPU ranks send commands, they do not receive them."));

    int message[2] = {0, 0};

    if (idle_sleep_microseconds() == 0)
      {
        const int ierr = MPI_Bcast(message, 2, MPI_INT, 0, layout.comm_grace);
        AssertThrowMPI(ierr);
      }
    else
      {
        MPI_Request request;
        int ierr = MPI_Irecv(message, 2, MPI_INT, 0, command_tag, layout.comm_grace, &request);
        AssertThrowMPI(ierr);

        const std::chrono::microseconds sleep_time(idle_sleep_microseconds());
        while (true)
          {
            int received = 0;
            ierr         = MPI_Test(&request, &received, MPI_STATUS_IGNORE);
            AssertThrowMPI(ierr);
            if (received != 0)
              break;
            std::this_thread::sleep_for(sleep_time);
          }
      }

    return {static_cast<Command>(message[0]), static_cast<unsigned int>(message[1])};
  }



  // CPU-only ranks: execute commands until stop. The GPU rank calls the same
  // function of its own cpu_side right after sending the command, so all
  // ranks of comm_world enter it together.
  template <typename CpuSide>
  void
  run_service_loop(const MPILayout &layout, CpuSide &cpu_side)
  {
#ifdef __linux__
    // the default timer slack of 50 us would be added to every sleep
    if (idle_sleep_microseconds() > 0)
      prctl(PR_SET_TIMERSLACK, 1000UL);
#endif

    while (true)
      {
        const auto [command, payload] = receive_command(layout);

        if (command == Command::setup)
          cpu_side.setup(payload);
        else if (command == Command::solve)
          cpu_side.coarse_solve();
        else if (command == Command::setup_cpu_only)
          cpu_side.setup(payload, true);
        else if (command == Command::solve_cg)
          cpu_side.solve();
        else if (command == Command::benchmark_levels)
          cpu_side.benchmark_levels(payload);
        else if (command == Command::benchmark_matvec)
          cpu_side.benchmark_matvec(payload);
        else
          break;
      }
  }
} // namespace hybrid
