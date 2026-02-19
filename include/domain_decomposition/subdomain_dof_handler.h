#ifndef subdomain_dof_handler_h
#define subdomain_dof_handler_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include "domain_decomposition/subdomain_triangulation.h"


DEAL_II_NAMESPACE_OPEN

template <int dim>
struct SubdomainDoFInfo
{
  /*
     Local (serial) subdomain to global (distributed) DoFs map.
  */
  std::vector<unsigned int> local_to_global_dof_map;

  /*
      Global interface DoFs in the global domain numbering.
  */
  IndexSet interface_dofs_global;

  /*
      Subdomain interface DoFs in the local subdomain numbering.
  */
  std::vector<unsigned int> local_interface_dofs;

  /*
      Subdomain interface DoFs in the global domain numbering.
  */
  std::vector<types::global_dof_index> interface_local_to_global_map;

  /*
      Local subdomain interface DoFs in the global interface numbering.
  */
  std::vector<unsigned int> subdomain_to_global_interface_map;

  /*
      Global interface DoFs to the local interface numbering.
  */
  std::map<types::global_dof_index, unsigned int>
    global_to_sudomain_interface_map;

  /*
    Physical boundary DoFs in the local subdomain numbering.
  */
  std::vector<unsigned int> local_physical_boundary_dofs;

  /*
    Id's of the cells that contain faces on the interface.
  */
  std::vector<unsigned int> interface_cell_ids;


  void
  clear()
  {
    local_to_global_dof_map.clear();
    interface_dofs_global.clear();
    local_interface_dofs.clear();
    interface_local_to_global_map.clear();
    global_to_sudomain_interface_map.clear();
    local_physical_boundary_dofs.clear();
    interface_cell_ids.clear();
  }
};


template <int dim>
class SubdomainDoFHandler : public EnableObserverPointer
{
public:
  SubdomainDoFHandler();

  void
  reinit(const SubdomainTriangulation<dim> &subdomain_triangulation);

  void
  distribute_subdomain_dofs(const DoFHandler<dim> &distributed_dof_handler);

  const DoFHandler<dim> &
  get_dof_handler() const;

  const SubdomainDoFInfo<dim> &
  get_dof_info() const;

  unsigned int
  get_subdomain_id() const;

private:
  SubdomainDoFInfo<dim> subdomain_dof_info;

  DoFHandler<dim> subdomain_dof_handler;

  ObserverPointer<const SubdomainTriangulation<dim>> subdomain_triangulation;

  unsigned int subdomain_id;
};

template <int dim>
SubdomainDoFHandler<dim>::SubdomainDoFHandler()
  : subdomain_triangulation(nullptr)
{
  subdomain_dof_info.clear();
  subdomain_id = numbers::invalid_unsigned_int;
}

template <int dim>
void
SubdomainDoFHandler<dim>::reinit(
  const SubdomainTriangulation<dim> &subdomain_triangulation)

{
  this->subdomain_triangulation = &subdomain_triangulation;
  subdomain_dof_handler.reinit(subdomain_triangulation.get_triangulation());
  subdomain_dof_info.clear();
  subdomain_id = subdomain_triangulation.get_topology_info().subdomain_id;
}

template <int dim>
unsigned int
SubdomainDoFHandler<dim>::get_subdomain_id() const
{
  return subdomain_id;
}

template <int dim>
const DoFHandler<dim> &
SubdomainDoFHandler<dim>::get_dof_handler() const
{
  return subdomain_dof_handler;
}


template <int dim>
const SubdomainDoFInfo<dim> &
SubdomainDoFHandler<dim>::get_dof_info() const
{
  return subdomain_dof_info;
}


template <int dim>
void
SubdomainDoFHandler<dim>::distribute_subdomain_dofs(
  const DoFHandler<dim> &distributed_dof_handler)
{
  const auto subdomain_topology =
    this->subdomain_triangulation->get_topology_info();

  const auto &fe = distributed_dof_handler.get_fe();

  subdomain_dof_handler.distribute_dofs(fe);

  subdomain_dof_info.clear();

  subdomain_dof_info.local_to_global_dof_map.resize(
    subdomain_dof_handler.n_dofs());
  {
    auto global_cell     = distributed_dof_handler.begin_active();
    auto global_cell_end = distributed_dof_handler.end();

    auto local_cell = subdomain_dof_handler.begin_active();

    std::vector<types::global_dof_index> global_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (; global_cell != global_cell_end; ++global_cell)
      {
        if (global_cell->is_locally_owned())
          {
            global_cell->get_dof_indices(global_dof_indices);
            local_cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              {
                subdomain_dof_info
                  .local_to_global_dof_map[local_dof_indices[i]] =
                  global_dof_indices[i];
              }

            ++local_cell;
          }
      }
  }

  subdomain_dof_info.interface_dofs_global.set_size(
    distributed_dof_handler.n_dofs());

  subdomain_dof_info.subdomain_to_global_interface_map.resize(
    subdomain_dof_handler.n_dofs(), numbers::invalid_unsigned_int);

  IndexSet local_physical_boundary_dofs(subdomain_dof_handler.n_dofs());
  IndexSet local_interface_dofs(subdomain_dof_handler.n_dofs());

  const unsigned int n_dofs_per_cell = fe.dofs_per_cell;

  std::vector<types::global_dof_index> cell_dofs(n_dofs_per_cell);

  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      if (!cell->at_boundary())
        continue;

      cell->get_dof_indices(cell_dofs);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (cell->at_boundary(f) &&
              cell->face(f)->boundary_id() != subdomain_topology.interface_id)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                {
                  if (fe.has_support_on_face(i, f))
                    local_physical_boundary_dofs.add_index(cell_dofs[i]);
                }
            }
        }
    }

  unsigned int interface_cell_counter = 0;
  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      if (cell->at_boundary())
        {
          cell->get_dof_indices(cell_dofs);

          bool visited_interface_face = false;

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->at_boundary(f) && cell->face(f)->boundary_id() ==
                                            subdomain_topology.interface_id)
                {
                  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                    {
                      if (fe.has_support_on_face(i, f))
                        local_interface_dofs.add_index(cell_dofs[i]);
                    }

                  if (!visited_interface_face)
                    {
                      subdomain_dof_info.interface_cell_ids.push_back(
                        interface_cell_counter);

                      visited_interface_face = true;
                    }
                }
            }
        }
      ++interface_cell_counter;
    }

  local_interface_dofs.subtract_set(local_physical_boundary_dofs);

  for (auto index : local_physical_boundary_dofs)
    {
      subdomain_dof_info.local_physical_boundary_dofs.push_back(index);
    }

  subdomain_dof_info.subdomain_to_global_interface_map.resize(
    local_interface_dofs.size(), numbers::invalid_unsigned_int);

  unsigned int interface_counter = 0;
  for (auto index : local_interface_dofs)
    {
      const types::global_dof_index global_index =
        subdomain_dof_info.local_to_global_dof_map[index];


      subdomain_dof_info.local_interface_dofs.push_back(index);
      subdomain_dof_info.interface_local_to_global_map.push_back(global_index);
      subdomain_dof_info.interface_dofs_global.add_index(global_index);

      subdomain_dof_info.subdomain_to_global_interface_map[index] =
        interface_counter;

      subdomain_dof_info.global_to_sudomain_interface_map[global_index] =
        interface_counter;

      interface_counter++;
    }

  subdomain_dof_info.interface_dofs_global.compress();
}

DEAL_II_NAMESPACE_CLOSE

#endif
