#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <Kokkos_Core.hpp>

#include <sched.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace hybrid
{
  using namespace dealii;

  namespace internal
  {
    // CPU socket of the cores this rank may run on, -1 if they span several
    // sockets (rank not pinned) or the topology is unknown.
    inline int
    socket_of_affinity_mask()
    {
      // affinity mask = set of cores the OS may run this rank on
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        return -1;

      // Linux lists the socket of each core in sysfs; all allowed cores
      // have to agree
      int socket = -1;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask))
          {
            std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                               "/topology/physical_package_id");
            int cpu_socket = -1;
            file >> cpu_socket;
            if (cpu_socket < 0 || (socket >= 0 && cpu_socket != socket))
              return -1;
            socket = cpu_socket;
          }
      return socket;
    }



    // Number of hardware threads this rank may run on; 1 = pinned to one
    // core, 2 = pinned to one hyperthreaded core.
    inline int
    n_allowed_cpus()
    {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        return -1;
      return CPU_COUNT(&mask);
    }
  } // namespace internal



  /**
   * Rank roles for the hybrid CPU-GPU multigrid. The ranks of a node are
   * grouped into blocks of k = ranks_per_gpu consecutive node-local ranks, one
   * block per GPU (on JUPITER: per Grace). The first rank of a block drives
   * the GPU, the other k-1 ranks only work on the CPU levels.
   *
   *   comm_world: all ranks              -> CPU levels
   *   comm_gpu:   first rank of a block  -> GPU levels, outer CG
   *               (MPI_COMM_NULL on CPU-only ranks)
   *   comm_grace: the k ranks of a block -> commands, switch-level data
   *
   * Setting HYBRID_SIMULATED_RANKS_PER_NODE splits comm_world into fake nodes
   * of that many consecutive ranks, to mimic several nodes on one machine.
   */
  class MPILayout
  {
  public:
    // Collective on comm_world.
    MPILayout(const MPI_Comm comm_world, const unsigned int ranks_per_gpu);

    // Frees the communicators created here (not comm_world).
    ~MPILayout();

    // Owns MPI communicators: a copy would free them twice.
    MPILayout(const MPILayout &) = delete;
    MPILayout &
    operator=(const MPILayout &) = delete;

    // Collective on comm_world, output on rank 0.
    void
    print(std::ostream &out) const;

    // Collective on comm_world. False if the ranks of some block are not all
    // pinned to the same CPU socket.
    bool
    blocks_share_socket() const;

    // Collective on comm_world. False if SLURM_GPUS_ON_NODE is set and differs
    // from the number of blocks on some node (--ntasks-per-node != GPUs * k).
    bool
    blocks_match_slurm_gpus() const;

    const MPI_Comm     comm_world;    // all ranks, owned by the caller
    const unsigned int ranks_per_gpu; // k

    MPI_Comm comm_node  = MPI_COMM_NULL; // ranks on the same (possibly simulated) node
    MPI_Comm comm_grace = MPI_COMM_NULL; // the k ranks of this rank's block
    MPI_Comm comm_gpu   = MPI_COMM_NULL; // GPU ranks; MPI_COMM_NULL on CPU-only ranks

    bool         is_gpu_rank     = false; // first rank of its block
    bool         simulated_nodes = false;
    unsigned int node_index      = 0; // only for printing
    unsigned int grace_index     = 0; // == comm_gpu rank of the block's GPU rank
    unsigned int n_graces        = 0; // == number of GPUs == size of comm_gpu
  };



  // All communicators come from MPI_Comm_split(parent, color, key, &new):
  // collective on parent; same color -> same new communicator, ordered by
  // key; color MPI_UNDEFINED -> new = MPI_COMM_NULL.
  inline MPILayout::MPILayout(const MPI_Comm comm_world, const unsigned int ranks_per_gpu)
    : comm_world(comm_world)
    , ranks_per_gpu(ranks_per_gpu)
  {
    AssertThrow(ranks_per_gpu > 0, ExcMessage("ranks_per_gpu has to be positive."));

    const unsigned int world_rank = Utilities::MPI::this_mpi_process(comm_world);

    // 1. comm_node: ranks that share memory (= same node), or fake nodes of
    //    ranks_per_node consecutive ranks
    int ierr;
    if (const char *env = std::getenv("HYBRID_SIMULATED_RANKS_PER_NODE"))
      {
        const int ranks_per_node = std::atoi(env);
        AssertThrow(ranks_per_node > 0,
                    ExcMessage("HYBRID_SIMULATED_RANKS_PER_NODE has to be positive."));
        simulated_nodes = true;
        ierr = MPI_Comm_split(comm_world, world_rank / ranks_per_node, world_rank, &comm_node);
      }
    else
      ierr = MPI_Comm_split_type(
        comm_world, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &comm_node);
    AssertThrowMPI(ierr);

    const unsigned int node_rank = Utilities::MPI::this_mpi_process(comm_node);
    const unsigned int node_size = Utilities::MPI::n_mpi_processes(comm_node);

    // 2. k has to divide every node. min over 1/0 = logical AND, so all ranks
    //    throw together and no rank is left waiting in a later split
    const bool all_nodes_divisible =
      Utilities::MPI::min(node_size % ranks_per_gpu == 0 ? 1u : 0u, comm_world) == 1u;
    AssertThrow(all_nodes_divisible,
                ExcMessage("The number of ranks on every node has to be a multiple of "
                           "ranks_per_gpu = " +
                           std::to_string(ranks_per_gpu) + "."));

    // 3. comm_grace: blocks of k consecutive node ranks. Split from comm_node,
    //    so a block never spans two nodes. Block rank 0 drives the GPU.
    ierr = MPI_Comm_split(comm_node, node_rank / ranks_per_gpu, node_rank, &comm_grace);
    AssertThrowMPI(ierr);
    is_gpu_rank = Utilities::MPI::this_mpi_process(comm_grace) == 0;

    // 4. comm_gpu: GPU ranks only, kept in world order
    ierr = MPI_Comm_split(comm_world, is_gpu_rank ? 0 : MPI_UNDEFINED, world_rank, &comm_gpu);
    AssertThrowMPI(ierr);

    // 5. node_index: rank among the first ranks of all nodes, sent to the
    //    rest of the node
    MPI_Comm comm_node_leaders;
    ierr = MPI_Comm_split(
      comm_world, node_rank == 0 ? 0 : MPI_UNDEFINED, world_rank, &comm_node_leaders);
    AssertThrowMPI(ierr);
    if (node_rank == 0)
      {
        node_index = Utilities::MPI::this_mpi_process(comm_node_leaders);
        Utilities::MPI::free_communicator(comm_node_leaders);
      }
    node_index = Utilities::MPI::broadcast(comm_node, node_index, 0);

    // 6. grace_index: only the GPU rank knows it (its comm_gpu rank), so it
    //    sends it to its block
    if (is_gpu_rank)
      grace_index = Utilities::MPI::this_mpi_process(comm_gpu);
    grace_index = Utilities::MPI::broadcast(comm_grace, grace_index, 0);

    // 7. one block per GPU
    n_graces = Utilities::MPI::n_mpi_processes(comm_world) / ranks_per_gpu;
  }



  inline MPILayout::~MPILayout()
  {
    if (comm_gpu != MPI_COMM_NULL)
      Utilities::MPI::free_communicator(comm_gpu);
    Utilities::MPI::free_communicator(comm_grace);
    Utilities::MPI::free_communicator(comm_node);
  }



  inline bool
  MPILayout::blocks_share_socket() const
  {
    // socket of every rank in my block
    const std::vector<int> sockets =
      Utilities::MPI::all_gather(comm_grace, internal::socket_of_affinity_mask());

    // my block is fine if all its ranks report the same, known socket
    bool block_ok = sockets[0] >= 0;
    for (const int socket : sockets)
      block_ok = block_ok && socket == sockets[0];

    // logical AND over all blocks
    return Utilities::MPI::min(block_ok ? 1u : 0u, comm_world) == 1u;
  }



  inline bool
  MPILayout::blocks_match_slurm_gpus() const
  {
    // set by SLURM when GPUs are allocated; unset on pc02 -> nothing to check
    const char        *gpus_on_node   = std::getenv("SLURM_GPUS_ON_NODE");
    const unsigned int blocks_on_node = Utilities::MPI::n_mpi_processes(comm_node) / ranks_per_gpu;

    const bool node_ok =
      gpus_on_node == nullptr || std::atoi(gpus_on_node) == static_cast<int>(blocks_on_node);

    return Utilities::MPI::min(node_ok ? 1u : 0u, comm_world) == 1u;
  }



  inline void
  MPILayout::print(std::ostream &out) const
  {
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);

    // set by srun --gpu-bind on JUPITER, unset on pc02
    const char *visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");

    // GPU that Kokkos uses on this rank; -1 = no GPU backend (pc02)
#if KOKKOS_VERSION >= 40500
    const int kokkos_device = Kokkos::device_id();
#else
    const int kokkos_device = -2;
#endif

    // one table row per rank; cpu = core it runs on at this moment
    std::ostringstream line;
    line << std::setw(6) << Utilities::MPI::this_mpi_process(comm_world) << std::setw(6)
         << node_index << std::setw(7) << grace_index << std::setw(6)
         << (is_gpu_rank ? "GPU" : "CPU") << std::setw(10)
         << (is_gpu_rank ? std::to_string(Utilities::MPI::this_mpi_process(comm_gpu)) : "-")
         << std::setw(12) << Utilities::MPI::this_mpi_process(comm_grace) << std::setw(6)
         << sched_getcpu() << std::setw(11) << internal::socket_of_affinity_mask() << std::setw(9)
         << internal::n_allowed_cpus() << std::setw(8) << kokkos_device << "  "
         << (visible_devices ? visible_devices : "-") << "  " << host;

    // collect all rows on rank 0, which prints them in world-rank order
    const std::vector<std::string> lines = Utilities::MPI::gather(comm_world, line.str(), 0);

    if (Utilities::MPI::this_mpi_process(comm_world) == 0)
      {
        out << "Hybrid MPI layout: " << Utilities::MPI::n_mpi_processes(comm_world)
            << " ranks, " << ranks_per_gpu << " ranks per GPU, " << n_graces << " GPUs"
            << (simulated_nodes ? " (simulated nodes)" : "") << std::endl
            << " world  node  grace  role  gpu_rank  grace_rank   cpu     socket  n_cpus  device"
            << "  CUDA_VISIBLE_DEVICES  host" << std::endl;
        for (const std::string &l : lines)
          out << l << std::endl;
        out << std::endl;
      }
  }
} // namespace hybrid
