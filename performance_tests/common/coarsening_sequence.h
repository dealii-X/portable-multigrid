#ifndef coarsening_sequence_h
#define coarsening_sequence_h

#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <memory>
#include <vector>

namespace multigrid
{
  // Drop-in for create_geometric_coarsening_sequence() on globally refined
  // meshes: level l is the coarse grid refined l times, same policy.
  template <int dim>
  std::vector<std::shared_ptr<const dealii::Triangulation<dim>>>
  create_refined_coarsening_sequence(
    const dealii::parallel::distributed::Triangulation<dim> &tria,
    const dealii::RepartitioningPolicyTools::Base<dim>      &policy)
  {
    using namespace dealii;

    const unsigned int n_levels = tria.n_global_levels();
    const auto         n_cells_on_finest =
      static_cast<types::global_cell_index>(tria.n_cells(0)) *
      Utilities::pow<types::global_cell_index>(2, dim * (n_levels - 1));
    if (tria.n_global_active_cells() != n_cells_on_finest)
      return MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria, policy);

    auto [vertices, cells, subcell_data] = GridTools::get_coarse_mesh_description(tria);
    GridTools::delete_unused_vertices(vertices, cells, subcell_data);

    std::vector<std::shared_ptr<const Triangulation<dim>>> levels(n_levels);
    levels.back().reset(&tria, [](auto *) {});

    for (unsigned int l = 0; l + 1 < n_levels; ++l)
      {
        parallel::distributed::Triangulation<dim> level_tria(tria.get_mpi_communicator());
        for (const auto i : tria.get_manifold_ids())
          if (i != numbers::flat_manifold_id)
            level_tria.set_manifold(i, tria.get_manifold(i));
        level_tria.create_triangulation(vertices, cells, subcell_data);
        level_tria.refine_global(l);

        const auto description =
          TriangulationDescription::Utilities::create_description_from_triangulation(
            level_tria, policy.partition(level_tria));

        auto level =
          std::make_shared<parallel::fullydistributed::Triangulation<dim>>(
            tria.get_mpi_communicator());
        for (const auto i : tria.get_manifold_ids())
          if (i != numbers::flat_manifold_id)
            level->set_manifold(i, tria.get_manifold(i));
        level->create_triangulation(description);
        levels[l] = level;
      }

    return levels;
  }
} // namespace multigrid

#endif
