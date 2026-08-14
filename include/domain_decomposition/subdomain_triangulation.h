#ifndef subdomain_triangulation_h
#define subdomain_triangulation_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>


DEAL_II_NAMESPACE_OPEN

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

  void
  copy_subdomain_triangulation(const SubdomainTriangulation<dim> &other_subdomain_triangulation);

  const Triangulation<dim> &
  get_triangulation() const;

  unsigned int
  get_subdomain_id() const;

  unsigned int
  get_interface_id() const;

  void
  refine_global(unsigned int n_refinement_cycles);


private:
  Triangulation<dim> subdomain_triangulation;

  unsigned int subdomain_id;

  unsigned int interface_id;
};

template <int dim>
SubdomainTriangulation<dim>::SubdomainTriangulation()
{
  this->clear();
  this->subdomain_id = numbers::invalid_unsigned_int;
  this->interface_id = numbers::invalid_unsigned_int;
}

template <int dim>
void
SubdomainTriangulation<dim>::clear()
{
  subdomain_triangulation.clear();
}

template <int dim>
const Triangulation<dim> &
SubdomainTriangulation<dim>::get_triangulation() const
{
  return subdomain_triangulation;
}

template <int dim>
void
SubdomainTriangulation<dim>::refine_global(unsigned int n_refinement_cycles)
{
  subdomain_triangulation.refine_global(n_refinement_cycles);
}

template <int dim>
unsigned int
SubdomainTriangulation<dim>::get_subdomain_id() const
{
  return this->subdomain_id;
}

template <int dim>
unsigned int
SubdomainTriangulation<dim>::get_interface_id() const
{
  return this->interface_id;
}

template <int dim>
template <typename TriaType>
void
SubdomainTriangulation<dim>::create_subdomain_triangulation(TriaType &distributed_triangulation)
{
  this->clear();

  this->subdomain_id =
    Utilities::MPI::this_mpi_process(distributed_triangulation.get_mpi_communicator());
  this->interface_id = 100 + this->subdomain_id;


  std::vector<CellData<dim>> subdomain_cell_data;
  SubCellData                subcell_data;
  std::vector<bool>          is_physical_boundary;

  std::map<unsigned int, unsigned int> global_to_local_vertex_map;

  std::vector<Point<dim>> vertices;
  std::vector<bool>       interface_vertex_ids;
  std::vector<bool>       physical_boundary_vertex_ids;

  for (const auto &cell : distributed_triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          CellData<dim> cell_data;
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            {
              const unsigned int global_vertex_index = cell->vertex_index(v);

              if (global_to_local_vertex_map.find(global_vertex_index) ==
                  global_to_local_vertex_map.end())
                {
                  global_to_local_vertex_map[global_vertex_index] = vertices.size();
                  vertices.push_back(cell->vertex(v));
                }
              cell_data.vertices[v] = global_to_local_vertex_map[global_vertex_index];
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
                  for (unsigned int fv = 0; fv < GeometryInfo<dim>::vertices_per_face; ++fv)
                    face_data.vertices[fv] =
                      global_to_local_vertex_map[cell->face(f)->vertex_index(fv)];

                  face_data.boundary_id =
                    on_physical_boundary ? cell->face(f)->boundary_id() : this->interface_id;

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

  this->subdomain_triangulation.create_triangulation(vertices, subdomain_cell_data, subcell_data);

  AssertDimension(this->subdomain_triangulation.n_active_cells(),
                  distributed_triangulation.n_locally_owned_active_cells());
}

template <int dim>
void
SubdomainTriangulation<dim>::copy_subdomain_triangulation(
  const SubdomainTriangulation<dim> &other_subdomain_triangulation)
{
  this->subdomain_triangulation.copy_triangulation(
    other_subdomain_triangulation.get_triangulation());
  this->interface_id = other_subdomain_triangulation.interface_id;
  this->subdomain_id = other_subdomain_triangulation.subdomain_id;
}

DEAL_II_NAMESPACE_CLOSE

#endif
