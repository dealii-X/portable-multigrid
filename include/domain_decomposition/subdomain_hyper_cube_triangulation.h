#ifndef subdomain_hyper_cube_triangulation_h
#define subdomain_hyper_cube_triangulation_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include "domain_decomposition/subdomain_triangulation.h"

DEAL_II_NAMESPACE_OPEN

template <int dim>
class SubdomainHyperCubeTriangulation : public EnableObserverPointer
{
public:
  SubdomainHyperCubeTriangulation(MPI_Comm mpi_communicator);

  void
  clear();

  // void
  // generate_fully_distributed_triangulation(
  //   unsigned int n_refinement_cycles = 0);

  void
  generate_subdomain_triangulations(
    unsigned int n_cell_per_subdomain_direction = 0);

  void
  build_distributed_from_subdomain(
    unsigned int n_cells_per_subdomain_direction);

  // create_subdomain_triangulation(
  //   parallel::distributed::Triangulation<dim> &distributed_triangulation);


  const parallel::fullydistributed::Triangulation<dim> &
  get_distributed_triangulation() const;


  const SubdomainTriangulation<dim> &
  get_subdomain_triangulation() const;

  // const SubdomainTopologyInfo<dim> &
  // get_topology_info() const;

  void
  save_triangulations() const;

  void
  refine_global(unsigned int n_refinement_cycles);


private:
  // SubdomainTopologyInfo<dim> topology_info;

  MPI_Comm mpi_communicator;

  Triangulation<dim> coarse_triangulation;

  SubdomainTriangulation<dim> subdomain_triangulation;

  parallel::fullydistributed::Triangulation<dim> distributed_triangulation;

  const unsigned int n_subdomains;

  const unsigned int this_subdomain_id;

  // const unsigned int n_cell_per_subdomain;
};

template <int dim>
SubdomainHyperCubeTriangulation<dim>::SubdomainHyperCubeTriangulation(
  MPI_Comm mpi_communicator)
  : mpi_communicator(mpi_communicator)
  , distributed_triangulation(mpi_communicator)
  , n_subdomains(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_subdomain_id(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  bool is_power_of_2 = (n_subdomains & (n_subdomains - 1)) == 0;

  const unsigned int root     = std::round(std::pow(n_subdomains, 1.0 / dim));
  unsigned int       root_pow = 1;
  for (int d = 0; d < dim; ++d)
    root_pow *= root;
  bool is_perfect_root = (root_pow == n_subdomains);

  AssertThrow(
    is_power_of_2 || is_perfect_root,
    StandardExceptions::ExcMessage(
      "n_ranks must be a power of 2 OR a perfect n^dim (e.g., 8, 9, 16, 27)."));

  this->clear();
}

template <int dim>
void
SubdomainHyperCubeTriangulation<dim>::clear()
{
  subdomain_triangulation.clear();
  distributed_triangulation.clear();
  coarse_triangulation.clear();
}


template <int dim>
void
SubdomainHyperCubeTriangulation<dim>::save_triangulations() const
{
  {
    std::ofstream output("Grid-local-rank-" +
                         std::to_string(this_subdomain_id) + ".vtk");
    GridOut       grid_out;
    grid_out.write_vtk(subdomain_triangulation.get_triangulation(), output);
  }

  {
    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(distributed_triangulation,
                                             "distributed.vtu");
  }
}

template <int dim>
const parallel::fullydistributed::Triangulation<dim> &
SubdomainHyperCubeTriangulation<dim>::get_distributed_triangulation() const
{
  return distributed_triangulation;
}


template <int dim>
const SubdomainTriangulation<dim> &
SubdomainHyperCubeTriangulation<dim>::get_subdomain_triangulation() const
{
  return subdomain_triangulation;
}

template <int dim>
void
SubdomainHyperCubeTriangulation<dim>::refine_global(
  unsigned int n_refinement_cycles)
{
  distributed_triangulation.refine_global(n_refinement_cycles);
  subdomain_triangulation.refine_global(n_refinement_cycles);
}

template <int dim>
void
SubdomainHyperCubeTriangulation<dim>::generate_subdomain_triangulations(
  unsigned int n_cells_per_subdomain_direction)
{
  // 1. Calculate how many subdomains we need per axis
  std::vector<unsigned int> subdomains_per_axis(dim);
  const unsigned int root = std::round(std::pow(n_subdomains, 1.0 / dim));

  if (std::pow(root, dim) == n_subdomains)
    {
      for (unsigned int d = 0; d < dim; ++d)
        subdomains_per_axis[d] = root;
    }
  else
    {
      const unsigned int total_bits = std::round(std::log2(n_subdomains));
      for (unsigned int d = 0; d < dim; ++d)
        {
          unsigned int bits =
            total_bits / dim + (d < (total_bits % dim) ? 1 : 0);
          subdomains_per_axis[d] = 1 << bits;
        }
    }


  auto get_global_v_id = [&](const Point<dim> &p) {
    unsigned int id         = 0;
    unsigned int multiplier = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        unsigned int total_cells_d =
          subdomains_per_axis[d] * n_cells_per_subdomain_direction;
        unsigned int v_idx = std::round(p[d] * total_cells_d);
        id += v_idx * multiplier;
        multiplier *= (total_cells_d + 1);
      }
    return id;
  };


  // 2. Create the Coarse Grid (1 cell per MPI Rank)
  // This is the "Skeleton" of your big mesh.
  Triangulation<dim> coarse_tria;
  Point<dim>         p_max;
  for (unsigned int d = 0; d < dim; ++d)
    p_max[d] = 1.0;

  GridGenerator::subdivided_hyper_rectangle(
    coarse_tria, subdomains_per_axis, Point<dim>(), p_max, true);

  // for (auto &cell : coarse_tria.active_cell_iterators())
  //   {
  //     cell->set_subdomain_id(cell_index);
  //     cell_index++;
  //   }


  TriangulationDescription::Description<dim> description;
  description.comm = mpi_communicator;

  description.coarse_cell_vertices = coarse_tria.get_vertices();

  description.cell_infos.resize(1); // One level (the fine one)

  unsigned int local_cell_index = 0;
  for (const auto &cell : coarse_tria.active_cell_iterators())
    {
      TriangulationDescription::CellData<dim> cd;

      // We must tell the CellData how it connects to the vertices.
      // Since your version uses 'CellId::binary_type id' for connectivity:
      // We often have to use a specific constructor or the 'coarse_cells'
      // vector.

      unsigned int global_cell_index =
        this_subdomain_id * coarse_tria.n_active_cells() +
        local_cell_index;

      // Binary type is usually an array of 4 unsigned ints (for dim=3)
      std::array<unsigned int, 4> id_data = {global_cell_index, 0, 0, 0};
      cd.id                               = id_data;

      // cd.id           = cell->id();
      cd.subdomain_id = this_subdomain_id;
      description.cell_infos[0].push_back(cd);

      // Map the cell connectivity into the description's coarse_cells
      dealii::CellData<dim> standard_cd;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        standard_cd.vertices[v] = get_global_v_id(cell->vertex(v));

      description.coarse_cells.push_back(standard_cd);

      local_cell_index++;
    }

    

  // 6. Stitching
  distributed_triangulation.create_triangulation(description);

  subdomain_triangulation.create_subdomain_triangulation(
    distributed_triangulation);
}

// 3. Partition the Coarse Grid
// Every rank must see the same partitioning to "stitch" correctly.
// unsigned int cell_index = 0;
// for (auto &cell : coarse_tria.active_cell_iterators())
//   {
//     cell->set_subdomain_id(cell_index);
//     cell_index++;
//   }

// 4. Global Refinement (The "Big Mesh" part)
// Since coarse_tria is small, we can refine it a few times on all ranks.
// This generates the hierarchy needed for the Description.
// coarse_tria.refine_global(n_refinement_cycles);

// 5. Create the Description using the deal.II Utility
// // This handles all the complex CellId::binary_type and cell_infos mapping.
// const auto description =
//   TriangulationDescription::Utilities::create_description_from_triangulation(
//     coarse_tria, mpi_communicator);

// // 6. Initialize the Fully Distributed Triangulation
// distributed_triangulation.create_triangulation(description);



// Triangulation<dim> subdomain_tria;

// // 7. Extract the Local Subdomain Mesh for Matrix-Free
// // We only want the cells belonging to 'this_subdomain_id'
// subdomain_tria.clear();

// // Create a copy of the cells belonging to our rank into a serial tria
// // This provides the structured local grid for BNN local solvers.
// GridTools::create_triangulation_with_removed_cells(
//   coarse_tria,
//   [&](const typename Triangulation<dim>::active_cell_iterator &cell) {
//     return cell->subdomain_id() != this_subdomain_id;
//   },
//   subdomain_tria);

// std::cout << "Rank " << this_subdomain_id
//           << " created local triangulation with "
//           << subdomain_tria.n_active_cells() << " cells." << std::endl;
// }

// template <int dim>
// void
// SubdomainHyperCubeTriangulation<dim>::generate_subdomain_triangulations(
//   unsigned int n_cells_per_subdomain_direction)
// {
//   std::vector<unsigned int> subdomains_per_axis(dim);

//   const unsigned int root     = std::round(std::pow(n_subdomains, 1.0 /
//   dim)); unsigned int       root_pow = 1; for (int d = 0; d < dim; ++d)
//     root_pow *= root;

//   if (root_pow == n_subdomains)
//     {
//       for (unsigned int d = 0; d < dim; ++d)
//         subdomains_per_axis[d] = root;
//     }
//   // Case B: Power of 2 (e.g., 8 ranks in 2D -> 4x2)
//   else
//     {
//       const unsigned int total_bits = std::round(std::log2(n_subdomains));
//       for (unsigned int d = 0; d < dim; ++d)
//         {
//           unsigned int bits =
//             total_bits / dim + (d < (total_bits % dim) ? 1 : 0);
//           subdomains_per_axis[d] = 1 << bits;
//         }
//     }

//   // Calculate local coordinates and bounding box
//   std::vector<unsigned int> this_subdomain_coordinates(dim);
//   Point<dim>                p_min, p_max;
//   unsigned int              temp_rank = this_subdomain_id;

//   for (unsigned int d = 0; d < dim; ++d)
//     {
//       this_subdomain_coordinates[d] = temp_rank % subdomains_per_axis[d];
//       temp_rank /= subdomains_per_axis[d];

//       p_min[d] = (double)this_subdomain_coordinates[d] /
//       subdomains_per_axis[d]; p_max[d] =
//         (double)(this_subdomain_coordinates[d] + 1) / subdomains_per_axis[d];
//     }

//   Triangulation<dim> subdomain_tria;

//   subdomain_triangulation.clear();
//   std::vector<unsigned int> repetitions(dim,
//   n_cells_per_subdomain_direction);
//   GridGenerator::subdivided_hyper_rectangle(
//     subdomain_tria, repetitions, p_min, p_max, /*colorize*/ true);



//   auto get_global_v_id = [&](const Point<dim> &p) {
//     unsigned int id         = 0;
//     unsigned int multiplier = 1;
//     for (unsigned int d = 0; d < dim; ++d)
//       {
//         // Total cells across the whole domain in this direction
//         unsigned int total_cells_d =
//           subdomains_per_axis[d] * n_cells_per_subdomain_direction;

//         // Snap the coordinate [0, 1] to the integer vertex index [0,
//         // total_cells_d]
//         unsigned int v_idx = std::round(p[d] * total_cells_d);

//         id += v_idx * multiplier;
//         multiplier *= (total_cells_d + 1);
//       }
//     return id;
//   };

// std::vector<CellData<dim>> local_cells_data;
// local_cells_data.reserve(subdomain_tria.n_active_cells());

// using DistCellData = TriangulationDescription::CellData<dim>;
// std::vector<DistCellData> local_cells_data;
// local_cells_data.reserve(subdomain_tria.n_active_cells());

// for (const auto &cell : subdomain_tria.active_cell_iterators())
//   {
//     DistCellData cd;
//     for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//       {
//         cd.vertices[v] = get_global_v_id(cell->vertex(v));
//       }

//     // Note: TriangulationDescription::CellData usually expects
//     subdomain_id
//     // to be explicitly set here.
//     cd.subdomain_id = this_subdomain_id;
//     local_cells_data.push_back(cd);
//   }

// using DistCellData = TriangulationDescription::CellData<dim>;
// std::vector<DistCellData> local_cells_data;
// local_cells_data.reserve(subdomain_triangulation.n_active_cells());

// unsigned int local_cell_index = 0;
// for (const auto &cell : subdomain_triangulation.active_cell_iterators())
//   {
//     DistCellData cd;

//     // 1. Set the IDs
//     cd.subdomain_id       = this_subdomain_id;
//     cd.level_subdomain_id = this_subdomain_id;

//     // 2. Set Boundary IDs (Optional, but good for BNN)
//     for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
//       if (cell->at_boundary(f))
//         cd.boundary_ids.push_back({f, cell->face(f)->boundary_id()});

//     // 3. The ID is often used by deal.II to verify global consistency
//     // For a single-level mesh, we can use the global index
//     // (You might need to convert your global_id to CellId::binary_type)

//     local_cells_data.push_back(cd);
//     local_cell_index++;
//   }
// for (const auto &cell : subdomain_tria.active_cell_iterators())
//   {
//     CellData<dim> cd;
//     for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//       {
//         // Map the physical location of the vertex to the global index
//         cd.vertices[v] = get_global_v_id(cell->vertex(v));
//       }
//     // Keep material_id or other flags if needed
//     cd.material_id = cell->material_id();
//     local_cells_data.push_back(cd);
//   }

// TriangulationDescription::Description<dim> description;
// description.comm                 = mpi_communicator;
// description.coarse_cell_vertices = subdomain_tria.get_vertices();
// // description.cells    = local_cells_data;
// description.cell_infos.resize(1);
// description.cell_infos[0] = local_cells_data; // Your cells with Global IDs

// distributed_triangulation.create_triangulation(description);
// }

// template <int dim>
// const Triangulation<dim> &
// SubdomainTriangulation<dim>::get_triangulation() const
// {
//   return subdomain_triangulation;
// }



// template <int dim>
// void
// SubdomainHyperCubeTriangulation<dim>::create_subdomain_triangulation()
// {
//   this->clear();
//   this->topology_info.subdomain_id = Utilities::MPI::this_mpi_process(
//     distributed_triangulation.get_mpi_communicator());

//   this->topology_info.interface_id = 100 + this->topology_info.subdomain_id;

//   std::vector<CellData<dim>> subdomain_cell_data;
//   SubCellData                subcell_data;
//   std::vector<bool>          is_physical_boundary;

//   std::map<unsigned int, unsigned int> global_to_local_vertex_map;

//   for (const auto &cell : distributed_triangulation.active_cell_iterators())
//     {
//       if (cell->is_locally_owned())
//         {
//           CellData<dim> cell_data;
//           for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
//                ++v)
//             {
//               const unsigned int global_vertex_index = cell->vertex_index(v);

//               if (global_to_local_vertex_map.find(global_vertex_index) ==
//                   global_to_local_vertex_map.end())
//                 {
//                   global_to_local_vertex_map[global_vertex_index] =
//                     this->topology_info.vertices.size();
//                   this->topology_info.vertices.push_back(cell->vertex(v));
//                 }
//               cell_data.vertices[v] =
//                 global_to_local_vertex_map[global_vertex_index];
//             }

//           cell_data.material_id = cell->material_id();
//           cell_data.manifold_id = cell->manifold_id();
//           subdomain_cell_data.push_back(cell_data);

//           for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
//           ++f)
//             {
//               bool on_physical_boundary = cell->at_boundary(f);
//               bool on_interface         = false;

//               if (!on_physical_boundary)
//                 {
//                   if (cell->neighbor(f)->is_ghost())
//                     on_interface = true;
//                 }
//               if (on_physical_boundary || on_interface)
//                 {
//                   CellData<dim - 1> face_data;
//                   for (unsigned int fv = 0;
//                        fv < GeometryInfo<dim>::vertices_per_face;
//                        ++fv)
//                     face_data.vertices[fv] =
//                       global_to_local_vertex_map[cell->face(f)->vertex_index(
//                         fv)];

//                   face_data.boundary_id = on_physical_boundary ?
//                                             cell->face(f)->boundary_id() :
//                                             this->topology_info.interface_id;

//                   face_data.manifold_id = cell->face(f)->manifold_id();

//                   if constexpr (dim == 2)
//                     subcell_data.boundary_lines.push_back(face_data);

//                   if constexpr (dim == 3)
//                     subcell_data.boundary_quads.push_back(face_data);

//                   is_physical_boundary.push_back(true);
//                 }
//             }
//         }
//     }

//   Assert(subcell_data.check_consistency(dim),
//          ExcMessage("Subcell data are not filled consistenly."));

//   GridTools::consistently_order_cells<dim>(subdomain_cell_data);

//   this->subdomain_triangulation.create_triangulation(
//     this->topology_info.vertices, subdomain_cell_data, subcell_data);

//   this->topology_info.physical_boundary_vertex_ids.resize(
//     this->subdomain_triangulation.n_vertices(), false);

//   for (auto &cell : this->subdomain_triangulation.active_cell_iterators())
//     {
//       if (!cell->is_locally_owned())
//         continue;

//       for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
//         {
//           if (!cell->at_boundary(f))
//             continue;

//           const auto bid = cell->face(f)->boundary_id();

//           if (bid != this->topology_info.interface_id)
//             {
//               for (unsigned int fv = 0;
//                    fv < GeometryInfo<dim>::vertices_per_face;
//                    ++fv)
//                 {
//                   const unsigned int vertex_idx =
//                     cell->face(f)->vertex_index(fv);
//                   this->topology_info.physical_boundary_vertex_ids[vertex_idx]
//                   =
//                     true;
//                 }
//             }
//         }
//     }

//   this->topology_info.interface_vertex_ids.resize(
//     subdomain_triangulation.n_vertices(), false);

//   for (auto &cell : this->subdomain_triangulation.active_cell_iterators())
//     {
//       if (!cell->is_locally_owned())
//         continue;

//       for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
//         {
//           if (!cell->at_boundary(f))
//             continue;

//           const auto bid = cell->face(f)->boundary_id();

//           if (bid == this->topology_info.interface_id)
//             {
//               for (unsigned int fv = 0;
//                    fv < GeometryInfo<dim>::vertices_per_face;
//                    ++fv)
//                 {
//                   const unsigned int vertex_idx =
//                     cell->face(f)->vertex_index(fv);

//                   if (!this->topology_info
//                          .physical_boundary_vertex_ids[vertex_idx])
//                     this->topology_info.interface_vertex_ids[vertex_idx] =
//                     true;
//                 }
//             }
//         }
//     }

//   AssertDimension(this->subdomain_triangulation.n_active_cells(),
//                   distributed_triangulation.n_locally_owned_active_cells());
// }

DEAL_II_NAMESPACE_CLOSE

#endif
