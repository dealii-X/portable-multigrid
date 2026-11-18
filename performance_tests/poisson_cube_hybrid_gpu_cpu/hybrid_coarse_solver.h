#pragma once

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <ostream>

#include "hybrid_cpu_side.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_service.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * Coarse solver of the GPU side, called by its V-cycle on the switch level
   * s-1 (GPU ranks only): copy the right-hand side to the host, let the
   * block's CPU-only ranks join, run one V-cycle on the CPU levels (P in and
   * out inside CpuSide::coarse_solve()), copy the correction back.
   */
  template <typename CpuSideType>
  class HybridCoarseSolver
    : public MGCoarseGridBase<
        LinearAlgebra::distributed::Vector<typename CpuSideType::vcycle_number,
                                           MemorySpace::Default>>
  {
  public:
    using Number     = typename CpuSideType::vcycle_number;
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    HybridCoarseSolver(const MPILayout &layout, CpuSideType &cpu_side)
      : layout(layout)
      , cpu_side(cpu_side)
    {}

    void
    operator()(const unsigned int, VectorType &dst, const VectorType &src) const override
    {
      std::vector<Number> &host_values = cpu_side.gpu_values();
      AssertDimension(host_values.size(), src.locally_owned_size());

      Timer time;

      // wake up the CPU-only ranks of the block
      send_command(layout, Command::solve);
      times[0] += time.wall_time();

      // device -> host, owned entries; same layout on both ends
      time.restart();
      Kokkos::deep_copy(host_view(host_values), device_view(src));
      times[1] += time.wall_time();

      // P down, V-cycle on the CPU levels (all ranks), P up
      time.restart();
      cpu_side.coarse_solve();
      times[2] += time.wall_time();

      // host -> device
      time.restart();
      Kokkos::deep_copy(device_view(dst), host_view(host_values));
      times[3] += time.wall_time();

      ++n_calls;
    }

    // Rank 0 (of comm_gpu): times since reset_times(), rank-local like the
    // per-level times of MultigridSolver.
    void
    print_times(std::ostream &out) const
    {
      if (Utilities::MPI::this_mpi_process(layout.comm_gpu) == 0)
        out << "==== switch, rank 0: " << n_calls << " calls, bcast " << times[0]
            << " device->host " << times[1] << " CPU part " << times[2] << " host->device "
            << times[3] << std::endl;
    }

    void
    reset_times()
    {
      times[0] = times[1] = times[2] = times[3] = 0.;
      n_calls                                   = 0;
    }

  private:
    using HostView =
      Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using DeviceView = Kokkos::View<Number *,
                                    MemorySpace::Default::kokkos_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static HostView
    host_view(std::vector<Number> &values)
    {
      return HostView(values.data(), values.size());
    }

    // locally owned entries of a level vector
    static DeviceView
    device_view(const VectorType &v)
    {
      return DeviceView(const_cast<Number *>(v.get_values()), v.locally_owned_size());
    }

    const MPILayout &layout;
    CpuSideType     &cpu_side;

    mutable double times[4] = {0., 0., 0., 0.}; // bcast, device->host, CPU part, host->device
    mutable unsigned int n_calls = 0;
  };
} // namespace hybrid
