#pragma once

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id_translator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "hybrid_cell_id_translator.h"
#include "hybrid_mpi_layout.h"

namespace hybrid
{
  using namespace dealii;

  // Coarse grid of both sides. Building it with the same recipe on each
  // communicator gives the same cells with the same cell ids.
  // stretch < 0: n_subdivisions^dim cube on [-0.9, 1.0] (non-doubling mesh of
  // poisson_cube_*). stretch = 0, 1, 2: the doubling mesh, one cell of
  // [-1, 0.9]^dim with `stretch` of the directions doubled (2 cells on
  // [-1, 2.8]); with n_refine = cycle / 3, stretch = cycle % 3 it is cycle
  // `cycle` of poisson_cube_*'s doubling sequence.
  struct CoarseGrid
  {
    unsigned int n_subdivisions = 3;
    int          stretch        = -1;
  };

  template <int dim>
  void
  make_coarse_grid(Triangulation<dim> &tria, const CoarseGrid &grid)
  {
    if (grid.stretch < 0)
      {
        GridGenerator::subdivided_hyper_cube(tria, grid.n_subdivisions, -0.9, 1.0);
        return;
      }

    Point<dim>                p1, p2;
    std::vector<unsigned int> subdivisions(dim, 1);
    for (unsigned int d = 0; d < dim; ++d)
      {
        p1[d] = -1;
        p2[d] = d < static_cast<unsigned int>(grid.stretch) ? 2.8 : 0.9;
        if (d < static_cast<unsigned int>(grid.stretch))
          subdivisions[d] = 2;
      }
    GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, p1, p2);
  }



  // Polynomial degree of every level of the ph-hierarchy of poisson_cube_bk3:
  // Q1 on the n_h_levels h-levels, then p-levels on the fine mesh with
  // degrees ..., fe_degree - 2, fe_degree.
  inline std::vector<unsigned int>
  ph_level_degrees(const unsigned int n_h_levels, const unsigned int fe_degree)
  {
    std::vector<unsigned int> p_degrees({fe_degree});
    while (p_degrees.back() > 1)
      p_degrees.push_back(std::max(p_degrees.back() - 2, 1u));
    std::reverse(p_degrees.begin(), p_degrees.end()); // 1, ..., fe_degree

    std::vector<unsigned int> degrees(n_h_levels, 1);
    degrees.insert(degrees.end(), p_degrees.begin() + 1, p_degrees.end());
    return degrees;
  }



  // Cells of the coarse grid in direction d.
  inline unsigned int
  n_coarse_cells(const CoarseGrid &grid, const unsigned int d)
  {
    if (grid.stretch < 0)
      return grid.n_subdivisions;
    return d < static_cast<unsigned int>(grid.stretch) ? 2 : 1;
  }



  // Global number of DoFs of every level of the ph-hierarchy on the coarse
  // grid refined up to n_refine times.
  template <int dim>
  std::vector<types::global_dof_index>
  ph_level_n_dofs(const CoarseGrid  &grid,
                  const unsigned int n_refine,
                  const unsigned int fe_degree)
  {
    const std::vector<unsigned int> degrees = ph_level_degrees(n_refine + 1, fe_degree);

    std::vector<types::global_dof_index> n_dofs(degrees.size(), 1);
    for (unsigned int l = 0; l < degrees.size(); ++l)
      for (unsigned int d = 0; d < dim; ++d)
        n_dofs[l] *=
          (types::global_dof_index(n_coarse_cells(grid, d)) << std::min(l, n_refine)) * degrees[l] +
          1;
    return n_dofs;
  }



  // Finest level with at most max_dofs DoFs (global count), level 0 if there
  // is none. The levels up to it run on the CPUs.
  inline unsigned int
  find_switch_level(const std::vector<types::global_dof_index> &n_dofs,
                    const types::global_dof_index               max_dofs)
  {
    unsigned int level = 0;
    while (level + 1 < n_dofs.size() && n_dofs[level + 1] <= max_dofs)
      ++level;
    return level;
  }



  // Mesh hierarchy of one side on its communicator, built as in
  // poisson_cube_*: levels[l] = coarse grid refined l times.
  template <int dim>
  struct MeshLevels
  {
    MeshLevels(const MPI_Comm     comm,
               const CoarseGrid  &coarse_grid,
               const unsigned int n_refinements)
      : fine(comm)
    {
      make_coarse_grid(fine, coarse_grid);
      fine.refine_global(n_refinements);
      levels = MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        fine, RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(16));
    }

    // levels.back() points to it without owning it, so it has to live as
    // long as levels
    parallel::distributed::Triangulation<dim> fine;

    std::vector<std::shared_ptr<const Triangulation<dim>>> levels;
  };



  namespace internal
  {
    template <int dim>
    unsigned int
    n_locally_owned_cells(const Triangulation<dim> &tria)
    {
      return std::count_if(tria.begin_active(),
                           typename Triangulation<dim>::active_cell_iterator(tria.end()),
                           [](const auto &cell) { return cell.is_locally_owned(); });
    }



    // CellIDTranslator indices of the locally owned cells.
    template <int dim>
    IndexSet
    owned_cell_indices(const Triangulation<dim>                     &tria,
                       const dealii::internal::CellIDTranslator<dim> &translator)
    {
      std::vector<types::global_cell_index> indices;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          indices.push_back(translator.translate(cell));
      std::sort(indices.begin(), indices.end());

      IndexSet index_set(translator.size());
      index_set.add_indices(indices.begin(), indices.end());
      return index_set;
    }
  } // namespace internal



  // Collective on comm. Prints, on rank 0 of comm, the number of cells of
  // every level and on how many ranks they live.
  template <int dim>
  void
  print_level_partitions(const std::vector<std::shared_ptr<const Triangulation<dim>>> &levels,
                         const MPI_Comm                                                 comm,
                         const std::string                                             &label,
                         std::ostream                                                  &out)
  {
    std::ostringstream table;
    table << label << " (" << Utilities::MPI::n_mpi_processes(comm) << " ranks)" << std::endl
          << "  level     cells  ranks_with_cells  min_cells  max_cells" << std::endl;

    for (unsigned int l = 0; l < levels.size(); ++l)
      {
        const unsigned int n_local = internal::n_locally_owned_cells(*levels[l]);

        // min over the ranks that own cells
        const unsigned int n_ranks_with_cells = Utilities::MPI::sum(n_local > 0 ? 1u : 0u, comm);
        const unsigned int min_cells =
          Utilities::MPI::min(n_local > 0 ? n_local : numbers::invalid_unsigned_int, comm);
        const unsigned int max_cells = Utilities::MPI::max(n_local, comm);

        table << std::setw(7) << l << std::setw(10) << levels[l]->n_global_active_cells()
              << std::setw(18) << n_ranks_with_cells << std::setw(11) << min_cells
              << std::setw(11) << max_cells << std::endl;
      }

    if (Utilities::MPI::this_mpi_process(comm) == 0)
      out << table.str() << std::endl;
  }



  // Collective on comm_world. For the cells that a rank owns on the CPU side
  // of the switch level, looks up which rank owns them on the GPU side, and
  // counts how many belong to the GPU rank of the same block (ideally all:
  // then moving the switch-level data never leaves the block).
  // gpu_switch is nullptr on CPU-only ranks. Returns the fraction of such
  // cells over all ranks.
  template <int dim>
  double
  check_switch_level_ownership(const MPILayout          &layout,
                               const Triangulation<dim> *gpu_switch,
                               const Triangulation<dim> &cpu_switch,
                               std::ostream             &out)
  {
    // one integer per cell id; same coarse grid and number of levels on both
    // sides -> same numbering
    const auto translator = make_cell_id_translator(cpu_switch);
    if (layout.is_gpu_rank)
      AssertThrow(make_cell_id_translator(*gpu_switch).size() == translator.size(),
                  ExcMessage("GPU and CPU switch levels have different cell numberings."));

    const IndexSet gpu_owned = layout.is_gpu_rank ?
                                 internal::owned_cell_indices(*gpu_switch, translator) :
                                 IndexSet(translator.size());
    const IndexSet cpu_owned = internal::owned_cell_indices(cpu_switch, translator);

    // world rank of the GPU-side owner of each of my CPU-side cells
    const std::vector<unsigned int> gpu_owners =
      Utilities::MPI::compute_index_owner(gpu_owned, cpu_owned, layout.comm_world);

    const unsigned int world_rank = Utilities::MPI::this_mpi_process(layout.comm_world);
    const unsigned int my_gpu_rank =
      Utilities::MPI::broadcast(layout.comm_grace, world_rank, 0); // world rank of block's GPU rank

    const unsigned int n_cpu_cells = cpu_owned.n_elements();
    const unsigned int n_gpu_cells = gpu_owned.n_elements();
    const unsigned int n_in_block =
      std::count(gpu_owners.begin(), gpu_owners.end(), my_gpu_rank);

    // every cell is owned exactly once on each side
    const types::global_cell_index n_cells = cpu_switch.n_global_active_cells();
    AssertThrow(Utilities::MPI::sum<types::global_cell_index>(n_cpu_cells, layout.comm_world) ==
                    n_cells &&
                  Utilities::MPI::sum<types::global_cell_index>(n_gpu_cells, layout.comm_world) ==
                    n_cells,
                ExcMessage("Switch-level cells are not owned exactly once on each side."));

    std::ostringstream line;
    line << std::setw(6) << world_rank << std::setw(7) << layout.grace_index << std::setw(6)
         << (layout.is_gpu_rank ? "GPU" : "CPU") << std::setw(11)
         << (layout.is_gpu_rank ? std::to_string(n_gpu_cells) : "-") << std::setw(11)
         << n_cpu_cells << std::setw(15) << n_in_block;

    const std::vector<std::string> lines =
      Utilities::MPI::gather(layout.comm_world, line.str(), 0);

    const double fraction =
      static_cast<double>(Utilities::MPI::sum<types::global_cell_index>(n_in_block,
                                                                        layout.comm_world)) /
      n_cells;

    if (world_rank == 0)
      {
        out << "Switch level: " << n_cells << " cells" << std::endl
            << " world  grace  role  gpu_cells  cpu_cells  of_which_own_gpu" << std::endl;
        for (const std::string &l : lines)
          out << l << std::endl;
        out << "Cells whose GPU-side owner is the GPU rank of the same block: "
            << 100. * fraction << " %" << std::endl
            << std::endl;
      }

    return fraction;
  }
} // namespace hybrid
