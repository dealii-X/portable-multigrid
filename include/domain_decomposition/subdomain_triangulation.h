#ifndef subdomain_triangulation_h
#define subdomain_triangulation_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

DEAL_II_NAMESPACE_OPEN


template <int dim>
struct SubdomainTopologyInfo
{
  unsigned int            subdomain_id;
  std::vector<Point<dim>> vertices;
  std::vector<bool>       interface_vertex_ids;
  std::vector<bool>       physical_boundary_vertex_ids;
  types::boundary_id      interface_id;

  void
  clear()
  {
    vertices.clear();
    interface_vertex_ids.clear();
    physical_boundary_vertex_ids.clear();
    subdomain_id = numbers::invalid_unsigned_int;
    interface_id = numbers::invalid_boundary_id;
  }
};

template <int dim>
class SubdomainTriangulation : public EnableObserverPointer
{
public:
  SubdomainTriangulation();

  void
  clear();

  template <typename TriaType>
  void
  create_subdomain_triangulation(TriaType &distributed_triangulation);

  // void
  // create_subdomain_triangulation(
  //   parallel::fullydistributed::Triangulation<dim>
  //   &distributed_triangulation);

  const Triangulation<dim> &
  get_triangulation() const;

  const SubdomainTopologyInfo<dim> &
  get_topology_info() const;

  void
  refine_global(unsigned int n_refinement_cycles);


private:
  SubdomainTopologyInfo<dim> topology_info;

  Triangulation<dim> subdomain_triangulation;
};

template <int dim>
SubdomainTriangulation<dim>::SubdomainTriangulation()
{
  this->clear();
}

template <int dim>
void
SubdomainTriangulation<dim>::clear()
{
  subdomain_triangulation.clear();
  topology_info.clear();
}

template <int dim>
const Triangulation<dim> &
SubdomainTriangulation<dim>::get_triangulation() const
{
  return subdomain_triangulation;
}


template <int dim>
const SubdomainTopologyInfo<dim> &
SubdomainTriangulation<dim>::get_topology_info() const
{
  return topology_info;
}

template <int dim>
void
SubdomainTriangulation<dim>::refine_global(unsigned int n_refinement_cycles)
{
  subdomain_triangulation.refine_global(n_refinement_cycles);
}

template <int dim>
template <typename TriaType>
void
SubdomainTriangulation<dim>::create_subdomain_triangulation(
  TriaType &distributed_triangulation)
{
  this->clear();
  this->topology_info.subdomain_id = Utilities::MPI::this_mpi_process(
    distributed_triangulation.get_mpi_communicator());

  this->topology_info.interface_id = 100 + this->topology_info.subdomain_id;

  std::vector<CellData<dim>> subdomain_cell_data;
  SubCellData                subcell_data;
  std::vector<bool>          is_physical_boundary;

  std::map<unsigned int, unsigned int> global_to_local_vertex_map;

  for (const auto &cell : distributed_triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          CellData<dim> cell_data;
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              const unsigned int global_vertex_index = cell->vertex_index(v);

              if (global_to_local_vertex_map.find(global_vertex_index) ==
                  global_to_local_vertex_map.end())
                {
                  global_to_local_vertex_map[global_vertex_index] =
                    this->topology_info.vertices.size();
                  this->topology_info.vertices.push_back(cell->vertex(v));
                }
              cell_data.vertices[v] =
                global_to_local_vertex_map[global_vertex_index];
            }

          cell_data.material_id = cell->material_id();
          cell_data.manifold_id = cell->manifold_id();
          subdomain_cell_data.push_back(cell_data);

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              bool on_physical_boundary = cell->at_boundary(f);
              bool on_interface         = false;

              if (!on_physical_boundary)
                {
                  if (cell->neighbor(f)->is_ghost())
                    on_interface = true;
                }
              if (on_physical_boundary || on_interface)
                {
                  CellData<dim - 1> face_data;
                  for (unsigned int fv = 0;
                       fv < GeometryInfo<dim>::vertices_per_face;
                       ++fv)
                    face_data.vertices[fv] =
                      global_to_local_vertex_map[cell->face(f)->vertex_index(
                        fv)];

                  face_data.boundary_id = on_physical_boundary ?
                                            cell->face(f)->boundary_id() :
                                            this->topology_info.interface_id;

                  face_data.manifold_id = cell->face(f)->manifold_id();

                  if constexpr (dim == 2)
                    subcell_data.boundary_lines.push_back(face_data);

                  if constexpr (dim == 3)
                    subcell_data.boundary_quads.push_back(face_data);

                  is_physical_boundary.push_back(true);
                }
            }
        }
    }

  Assert(subcell_data.check_consistency(dim),
         ExcMessage("Subcell data are not filled consistenly."));

  GridTools::consistently_order_cells<dim>(subdomain_cell_data);

  this->subdomain_triangulation.create_triangulation(
    this->topology_info.vertices, subdomain_cell_data, subcell_data);

  this->topology_info.physical_boundary_vertex_ids.resize(
    this->subdomain_triangulation.n_vertices(), false);

  for (auto &cell : this->subdomain_triangulation.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->at_boundary(f))
            continue;

          const auto bid = cell->face(f)->boundary_id();

          if (bid != this->topology_info.interface_id)
            {
              for (unsigned int fv = 0;
                   fv < GeometryInfo<dim>::vertices_per_face;
                   ++fv)
                {
                  const unsigned int vertex_idx =
                    cell->face(f)->vertex_index(fv);
                  this->topology_info.physical_boundary_vertex_ids[vertex_idx] =
                    true;
                }
            }
        }
    }

  this->topology_info.interface_vertex_ids.resize(
    subdomain_triangulation.n_vertices(), false);

  for (auto &cell : this->subdomain_triangulation.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->at_boundary(f))
            continue;

          const auto bid = cell->face(f)->boundary_id();

          if (bid == this->topology_info.interface_id)
            {
              for (unsigned int fv = 0;
                   fv < GeometryInfo<dim>::vertices_per_face;
                   ++fv)
                {
                  const unsigned int vertex_idx =
                    cell->face(f)->vertex_index(fv);

                  if (!this->topology_info
                         .physical_boundary_vertex_ids[vertex_idx])
                    this->topology_info.interface_vertex_ids[vertex_idx] = true;
                }
            }
        }
    }

  AssertDimension(this->subdomain_triangulation.n_active_cells(),
                  distributed_triangulation.n_locally_owned_active_cells());
}

DEAL_II_NAMESPACE_CLOSE

#endif
