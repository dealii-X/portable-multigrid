#ifndef subdomain_dof_handler_h
#define subdomain_dof_handler_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "domain_decomposition/subdomain_triangulation.h"


DEAL_II_NAMESPACE_OPEN

template <int dim>
struct SubdomainDoFInfo
{
  /**
   * All interface DoFs across all subdomains (MPI ranks) in the global domain numbering.
   */
  IndexSet all_interface_dofs_global;

  /**
   * Local (serial) subdomain to global (distributed) DoFs map.
   */
  std::vector<types::global_dof_index> subdomain_to_global_dof_map;

  /**
   * Subdomain interface DoFs in the global domain numbering.
   */
  IndexSet subdomain_interface_dofs_global;

  /**
   * Subdomain interior dofs (i.e. DoFs that are not at the interface or the
   * physical (eventually Dirichlet) boundary) in the local subdomain numbering.
   */
  IndexSet subdomain_interior_dofs;

  /**
   * Physical boundary DoFs in the local subdomain numbering.
   */
  IndexSet subdomain_physical_boundary_dofs;

  /**
   * Subdomain interface DoFs in the local subdomain numbering.
   */
  IndexSet subdomain_interface_dofs;

  /**
   * Subdomain interface DoFs in the global domain numbering.
   */
  std::vector<types::global_dof_index> interface_local_to_global_map;

  /**
   * Global interface DoFs to the subdomain interface numbering.
   */
  std::map<types::global_dof_index, unsigned int> global_to_subdomain_interface_map;

  /**
   * Id's of the cells that contain faces on the interface.
   */
  std::vector<unsigned int> interface_cell_ids;

  /**
   * Global interface vertex (corner) DoFs.
   */
  IndexSet interface_vertex_dofs_global;

  /**
   * Global interface edge DoFs.
   */
  IndexSet interface_edge_dofs_global;

  /**
   * Global interface face DoFs in 3d.
   */
  IndexSet interface_face_dofs_global;

  /**
   * Subdomain interface vertex (corner) DoFs.
   * first - global numbering, second- local subdomain numbering
   */
  std::vector<std::pair<types::global_dof_index, unsigned int>> subdomain_interface_vertex_dofs;

  /**
   * Subdomain interface edge DoFs.
   * first - global numbering, second- local subdomain numbering
   *
   */
  std::vector<std::pair<types::global_dof_index, unsigned int>> subdomain_interface_edge_dofs;

  /**
   * Subdomain interface face DoFs in 3d.
   * first - global numbering, second- local subdomain numbering
   *
   */
  std::vector<std::pair<types::global_dof_index, unsigned int>> subdomain_interface_face_dofs;

  void
  clear()
  {
    subdomain_to_global_dof_map.clear();
    subdomain_interface_dofs.clear();
    subdomain_interface_dofs_global.clear();
    interface_local_to_global_map.clear();
    global_to_subdomain_interface_map.clear();
    subdomain_physical_boundary_dofs.clear();
    interface_cell_ids.clear();
    subdomain_interior_dofs.clear();
    interface_vertex_dofs_global.clear();
    interface_edge_dofs_global.clear();
    interface_face_dofs_global.clear();
    subdomain_interface_vertex_dofs.clear();
    subdomain_interface_edge_dofs.clear();
    subdomain_interface_face_dofs.clear();
  }
};


template <int dim>
class SubdomainDoFHandler : public EnableObserverPointer
{
public:
  SubdomainDoFHandler();

  void
  reinit(const std::shared_ptr<const SubdomainTriangulation<dim>> subdomain_triangulation,
         const DoFHandler<dim>                                   &distributed_dof_handler);

  void
  distribute_subdomain_dofs();

  void
  categorize_interface_dofs(const std::vector<IndexSet> &all_interface_dof_sets);

  const DoFHandler<dim> &
  get_dof_handler() const;

  const SubdomainDoFInfo<dim> &
  get_dof_info() const;

  unsigned int
  get_subdomain_id() const;

  types::boundary_id
  get_interface_id() const;

  MPI_Comm
  get_mpi_communicator() const;

  template <typename VectorType>
  void
  initialize_interface_dof_vector(VectorType &vec) const;

  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_interface_vector_partitioner() const;

  unsigned int
  local_to_global_interface_partitioner(const unsigned int local_index) const;

  types::global_dof_index
  local_interface_to_subdomain(const unsigned int local_index) const;

  types::global_dof_index
  local_interface_to_global_interface(const unsigned int local_index) const;

  unsigned int
  n_locally_relevant_interface_indices() const;

  unsigned int
  n_subdomains() const;

private:
  void
  fill_dof_info();

  SubdomainDoFInfo<dim> subdomain_dof_info;

  DoFHandler<dim> subdomain_dof_handler;

  std::shared_ptr<const SubdomainTriangulation<dim>> subdomain_triangulation;
  ObserverPointer<const DoFHandler<dim>>             distributed_dof_handler;

  unsigned int       subdomain_id;
  types::boundary_id interface_id;

  std::shared_ptr<const Utilities::MPI::Partitioner> interface_vector_partitioner;

  std::vector<types::global_dof_index> interface_indices_partitioner_to_subdomain_numbering;

  std::vector<types::global_dof_index> interface_indices_partitioner_to_global_numbering;
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
  const std::shared_ptr<const SubdomainTriangulation<dim>> subdomain_triangulation,
  const DoFHandler<dim>                                   &distributed_dof_handler)

{
  this->subdomain_triangulation = subdomain_triangulation;
  this->distributed_dof_handler = &distributed_dof_handler;

  subdomain_dof_handler.reinit(subdomain_triangulation->get_triangulation());
  subdomain_dof_info.clear();

  interface_indices_partitioner_to_subdomain_numbering.clear();
  interface_indices_partitioner_to_global_numbering.clear();

  subdomain_id = subdomain_triangulation->get_subdomain_id();
  interface_id = subdomain_triangulation->get_interface_id();
}

template <int dim>
void
SubdomainDoFHandler<dim>::distribute_subdomain_dofs()
{
  this->subdomain_dof_handler.distribute_dofs(this->distributed_dof_handler->get_fe());

  this->fill_dof_info();

  const IndexSet &local_interface_indices = subdomain_dof_info.subdomain_interface_dofs_global;
  IndexSet       &all_interface_dofs      = subdomain_dof_info.all_interface_dofs_global;

  std::vector<IndexSet> all_sets =
    Utilities::MPI::all_gather(this->get_mpi_communicator(), local_interface_indices);

  all_interface_dofs.clear();
  all_interface_dofs.set_size(this->distributed_dof_handler->n_dofs());

  for (const auto &set : all_sets)
    {
      all_interface_dofs.add_indices(set);
    }
  all_interface_dofs.compress();

  if (all_interface_dofs.n_elements() == 0)
    {
      this->interface_vector_partitioner = nullptr;
      return;
    }
  const unsigned int n_global_interface_dofs = all_interface_dofs.n_elements();

  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());

  IndexSet locally_owned_interface_indices_global_numbering(
    this->distributed_dof_handler->n_dofs());

  for (const auto mesh_id : local_interface_indices)
    {
      std::vector<unsigned int> sharers;
      for (unsigned int r = 0; r < n_ranks; ++r)
        {
          if (all_sets[r].is_element(mesh_id))
            {
              sharers.push_back(r);
            }
        }
      std::sort(sharers.begin(), sharers.end());

      unsigned int master_rank = sharers[0];

      if (my_rank == master_rank)
        {
          locally_owned_interface_indices_global_numbering.add_index(mesh_id);
        }
    }

  IndexSet ghost_interface_indices_global_numbering(local_interface_indices);
  ghost_interface_indices_global_numbering.subtract_set(
    locally_owned_interface_indices_global_numbering);

  const unsigned int n_locally_owned =
    locally_owned_interface_indices_global_numbering.n_elements();

  std::vector<unsigned int> all_n_locally_owned =
    Utilities::MPI::all_gather(this->get_mpi_communicator(), n_locally_owned);

  std::vector<unsigned int> rank_offsets(n_ranks + 1, 0);
  for (unsigned int r = 0; r < n_ranks; ++r)
    rank_offsets[r + 1] = rank_offsets[r] + all_n_locally_owned[r];

  const unsigned int total_interface_size = rank_offsets.back();

  AssertDimension(total_interface_size, n_global_interface_dofs);

  std::map<types::global_dof_index, unsigned int> global_numbering_to_interface_partitoner;
  std::vector<types::global_dof_index>            interface_partitioner_to_global_numbering;

  std::vector<unsigned int> rank_counter_ptr = rank_offsets;

  for (unsigned int r = 0; r < n_ranks; ++r)
    {
      for (const auto global_index : all_sets[r])
        {
          unsigned int master_rank = n_ranks + 1;
          for (unsigned int s = 0; s < n_ranks; ++s)
            {
              if (all_sets[s].is_element(global_index))
                {
                  master_rank = s;
                  break;
                }
            }
          if (master_rank == r)
            {
              global_numbering_to_interface_partitoner[global_index] = rank_counter_ptr[r];

              interface_partitioner_to_global_numbering.push_back(global_index);

              ++rank_counter_ptr[r];
            }
        }
    }

  IndexSet locally_owned_interface_indices(total_interface_size);
  locally_owned_interface_indices.add_range(rank_offsets[my_rank], rank_offsets[my_rank + 1]);

  IndexSet ghost_interface_indices(total_interface_size);

  for (const auto &index : local_interface_indices)
    {
      if (ghost_interface_indices_global_numbering.is_element(index))
        ghost_interface_indices.add_index(global_numbering_to_interface_partitoner[index]);
    }

  this->interface_vector_partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(locally_owned_interface_indices,
                                                  ghost_interface_indices,
                                                  this->get_mpi_communicator());

  const unsigned int n_locally_relevant_interface_dofs =
    this->interface_vector_partitioner->locally_owned_size() +
    this->interface_vector_partitioner->n_ghost_indices();


  this->interface_indices_partitioner_to_subdomain_numbering.clear();
  this->interface_indices_partitioner_to_subdomain_numbering.resize(
    n_locally_relevant_interface_dofs);

  this->interface_indices_partitioner_to_global_numbering.clear();
  this->interface_indices_partitioner_to_global_numbering.resize(n_locally_relevant_interface_dofs);

  for (unsigned int i = 0; i < n_locally_relevant_interface_dofs; ++i)
    {
      const unsigned int global_index_partitioner =
        this->interface_vector_partitioner->local_to_global(i);

      const types::global_dof_index global_index =
        interface_partitioner_to_global_numbering[global_index_partitioner];

      const auto subdomain_index =
        subdomain_dof_info.global_to_subdomain_interface_map.at(global_index);

      this->interface_indices_partitioner_to_subdomain_numbering[i] = subdomain_index;

      this->interface_indices_partitioner_to_global_numbering[i] = global_index;
    }

  this->categorize_interface_dofs(all_sets);
}


template <int dim>
void
SubdomainDoFHandler<dim>::categorize_interface_dofs(
  const std::vector<IndexSet> &all_interface_dof_sets)
{
  subdomain_dof_info.interface_vertex_dofs_global.set_size(this->distributed_dof_handler->n_dofs());
  subdomain_dof_info.interface_edge_dofs_global.set_size(this->distributed_dof_handler->n_dofs());
  subdomain_dof_info.interface_face_dofs_global.set_size(this->distributed_dof_handler->n_dofs());

  const IndexSet    &all_interface_dofs      = subdomain_dof_info.all_interface_dofs_global;
  const unsigned int n_global_interface_dofs = all_interface_dofs.n_elements();

  std::map<std::set<unsigned int>, std::vector<std::pair<types::global_dof_index, unsigned int>>>
    equivalence_dof_classes;

  std::vector<std::set<unsigned int>> subdomains_per_dof(n_global_interface_dofs);

  for (unsigned int i = 0; i < n_global_interface_dofs; ++i)
    {
      const auto global_idx = all_interface_dofs.nth_index_in_set(i);

      for (unsigned int p = 0; p < this->n_subdomains(); ++p)
        {
          if (all_interface_dof_sets[p].is_element(global_idx))
            subdomains_per_dof[i].insert(p);
        }
    }

  // extract proper subsets: sort and keep unique only
  std::vector<std::set<unsigned int>> equivalent_dof_classes = subdomains_per_dof;
  std::sort(equivalent_dof_classes.begin(), equivalent_dof_classes.end());
  auto last = std::unique(equivalent_dof_classes.begin(), equivalent_dof_classes.end());
  equivalent_dof_classes.erase(last, equivalent_dof_classes.end());

  std::vector<std::vector<unsigned int>> dofs_per_equivalence_class(equivalent_dof_classes.size());

  for (unsigned int i = 0; i < n_global_interface_dofs; ++i)
    {
      auto it = std::lower_bound(equivalent_dof_classes.begin(),
                                 equivalent_dof_classes.end(),
                                 subdomains_per_dof[i]);

      unsigned int class_idx = std::distance(equivalent_dof_classes.begin(), it);

      dofs_per_equivalence_class[class_idx].push_back(all_interface_dofs.nth_index_in_set(i));
    }

  IndexSet used_dofs(this->distributed_dof_handler->n_dofs());

  // categorize inteface dofs
  for (unsigned int class_idx = 0; class_idx < equivalent_dof_classes.size(); ++class_idx)
    {
      const auto &class_set     = equivalent_dof_classes[class_idx];
      const auto &dofs_in_class = dofs_per_equivalence_class[class_idx];

      if (dofs_in_class.size() == 1 && class_set.size() > 2)
        {
          // vertex (corner) interface dofs are shared by a unique maximal set of processors (more
          // than 2)

          for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
            {
              const auto global_idx = dofs_in_class[i];

              subdomain_dof_info.interface_vertex_dofs_global.add_index(global_idx);
              used_dofs.add_index(global_idx);

              if (subdomain_dof_info.subdomain_interface_dofs_global.is_element(global_idx))
                subdomain_dof_info.subdomain_interface_vertex_dofs.push_back(
                  std::make_pair(global_idx,
                                 subdomain_dof_info.global_to_subdomain_interface_map[global_idx]));
            }
        }
      else if constexpr (dim == 2)
        {
          // in 2d all other interface dofs are edge dofs
          for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
            {
              const auto global_idx = dofs_in_class[i];

              subdomain_dof_info.interface_edge_dofs_global.add_index(global_idx);
              used_dofs.add_index(global_idx);


              if (subdomain_dof_info.subdomain_interface_dofs_global.is_element(global_idx))
                subdomain_dof_info.subdomain_interface_edge_dofs.push_back(
                  std::make_pair(global_idx,
                                 subdomain_dof_info.global_to_subdomain_interface_map[global_idx]));
            }
        }
      else if constexpr (dim == 3)
        {
          if ((class_set.size() == 2) && (dofs_in_class.size() > 1))
            {
              // face dofs in 3d are shared by exactly two subdomains
              // we also filter out case there's only one dof shared by two subdomains

              for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
                {
                  const auto global_idx = dofs_in_class[i];

                  subdomain_dof_info.interface_face_dofs_global.add_index(global_idx);
                  used_dofs.add_index(global_idx);


                  if (subdomain_dof_info.subdomain_interface_dofs_global.is_element(global_idx))
                    subdomain_dof_info.subdomain_interface_face_dofs.push_back(std::make_pair(
                      global_idx,
                      subdomain_dof_info.global_to_subdomain_interface_map[global_idx]));
                }
            }
          else
            {
              // all other interface dofs in 3d are edge dofs
              for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
                {
                  const auto global_idx = dofs_in_class[i];

                  subdomain_dof_info.interface_edge_dofs_global.add_index(global_idx);
                  used_dofs.add_index(global_idx);


                  if (subdomain_dof_info.subdomain_interface_dofs_global.is_element(global_idx))
                    subdomain_dof_info.subdomain_interface_edge_dofs.push_back(std::make_pair(
                      global_idx,
                      subdomain_dof_info.global_to_subdomain_interface_map[global_idx]));
                }
            }
        }
    }
    
  used_dofs.compress();

  AssertThrow(
    all_interface_dofs == used_dofs,
    ExcMessage(
      "SubdomainDoFHandler<dim>::categorize_interface_dofs() couldn't categorize some of the dofs."));

  subdomain_dof_info.interface_vertex_dofs_global.compress();
  subdomain_dof_info.interface_edge_dofs_global.compress();
  subdomain_dof_info.interface_face_dofs_global.compress();


  // if (this->get_subdomain_id() == 0)
  //   {
  //     std::cout << "Vertex dofs: ";
  //     for (const auto &index : subdomain_dof_info.interface_vertex_dofs_global)
  //       std::cout << index << "   ";
  //     std::cout << std::endl;
  //     std::cout << "Edge dofs: ";
  //     for (const auto &index : subdomain_dof_info.interface_edge_dofs_global)
  //       std::cout << index << "   ";
  //     std::cout << std::endl;
  //     std::cout << "Face dofs: ";
  //     for (const auto &index : subdomain_dof_info.interface_face_dofs_global)
  //       std::cout << index << "   ";
  //     std::cout << std::endl;
  //   }
}


template <int dim>
void
SubdomainDoFHandler<dim>::fill_dof_info()
{
  subdomain_dof_info.clear();

  subdomain_dof_info.subdomain_to_global_dof_map.resize(subdomain_dof_handler.n_dofs());

  const auto &fe = this->distributed_dof_handler->get_fe();

  const unsigned int n_dofs_per_cell = fe.dofs_per_cell;

  {
    auto global_cell     = this->distributed_dof_handler->begin_active();
    auto global_cell_end = this->distributed_dof_handler->end();

    auto local_cell = subdomain_dof_handler.begin_active();

    std::vector<types::global_dof_index> global_dof_indices(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    for (; global_cell != global_cell_end; ++global_cell)
      {
        if (global_cell->is_locally_owned())
          {
            global_cell->get_dof_indices(global_dof_indices);
            local_cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              {
                subdomain_dof_info.subdomain_to_global_dof_map[local_dof_indices[i]] =
                  global_dof_indices[i];
              }

            ++local_cell;
          }
      }
  }

  subdomain_dof_info.subdomain_interface_dofs_global.set_size(
    this->distributed_dof_handler->n_dofs());

  subdomain_dof_info.subdomain_interface_dofs.set_size(subdomain_dof_handler.n_dofs());

  subdomain_dof_info.subdomain_physical_boundary_dofs.set_size(subdomain_dof_handler.n_dofs());

  subdomain_dof_info.subdomain_interior_dofs.set_size(subdomain_dof_handler.n_dofs());

  std::vector<types::global_dof_index> cell_dofs(n_dofs_per_cell);

  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      if (!cell->at_boundary())
        continue;

      cell->get_dof_indices(cell_dofs);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (cell->at_boundary(f) &&
              cell->face(f)->boundary_id() != this->subdomain_triangulation->get_interface_id())
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                {
                  if (fe.has_support_on_face(i, f))
                    subdomain_dof_info.subdomain_physical_boundary_dofs.add_index(cell_dofs[i]);
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
              if (cell->at_boundary(f) &&
                  cell->face(f)->boundary_id() == this->subdomain_triangulation->get_interface_id())
                {
                  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                    {
                      if (fe.has_support_on_face(i, f))
                        subdomain_dof_info.subdomain_interface_dofs.add_index(cell_dofs[i]);
                    }

                  if (!visited_interface_face)
                    {
                      subdomain_dof_info.interface_cell_ids.push_back(interface_cell_counter);

                      visited_interface_face = true;
                    }
                }
            }
        }
      ++interface_cell_counter;
    }

  subdomain_dof_info.subdomain_interface_dofs.subtract_set(
    subdomain_dof_info.subdomain_physical_boundary_dofs);

  subdomain_dof_info.subdomain_interior_dofs.add_range(0, subdomain_dof_handler.n_dofs());
  subdomain_dof_info.subdomain_interior_dofs.subtract_set(
    subdomain_dof_info.subdomain_physical_boundary_dofs);
  subdomain_dof_info.subdomain_interior_dofs.subtract_set(
    subdomain_dof_info.subdomain_interface_dofs);

  {
    for (const auto &index : subdomain_dof_info.subdomain_interface_dofs)
      {
        const types::global_dof_index global_index =
          subdomain_dof_info.subdomain_to_global_dof_map[index];

        subdomain_dof_info.interface_local_to_global_map.push_back(global_index);

        subdomain_dof_info.subdomain_interface_dofs_global.add_index(global_index);

        subdomain_dof_info.global_to_subdomain_interface_map[global_index] = index;
      }

    subdomain_dof_info.subdomain_interface_dofs_global.compress();
  }
}

template <int dim>
types::global_dof_index
SubdomainDoFHandler<dim>::local_interface_to_subdomain(const unsigned int local_index) const
{
  return this->interface_indices_partitioner_to_subdomain_numbering[local_index];
}

template <int dim>
types::global_dof_index
SubdomainDoFHandler<dim>::local_interface_to_global_interface(const unsigned int local_index) const
{
  return this->interface_indices_partitioner_to_global_numbering[local_index];
}

template <int dim>
unsigned int
SubdomainDoFHandler<dim>::get_subdomain_id() const
{
  return subdomain_id;
}

template <int dim>
unsigned int
SubdomainDoFHandler<dim>::n_subdomains() const
{
  return Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
}

template <int dim>
types::boundary_id
SubdomainDoFHandler<dim>::get_interface_id() const
{
  return interface_id;
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
MPI_Comm
SubdomainDoFHandler<dim>::get_mpi_communicator() const
{
  return this->distributed_dof_handler->get_mpi_communicator();
}

template <int dim>
unsigned int
SubdomainDoFHandler<dim>::n_locally_relevant_interface_indices() const
{
  return this->interface_vector_partitioner->locally_owned_size() +
         this->interface_vector_partitioner->n_ghost_indices();
}

template <int dim>
unsigned int
SubdomainDoFHandler<dim>::local_to_global_interface_partitioner(
  const unsigned int local_index) const
{
  return this->interface_vector_partitioner->local_to_global(local_index);
}


template <int dim>
const std::shared_ptr<const Utilities::MPI::Partitioner> &
SubdomainDoFHandler<dim>::get_interface_vector_partitioner() const
{
  return this->interface_vector_partitioner;
}

template <int dim>
template <typename VectorType>
void
SubdomainDoFHandler<dim>::initialize_interface_dof_vector(VectorType &vec) const
{
  vec.reinit(this->interface_vector_partitioner);
}

DEAL_II_NAMESPACE_CLOSE

#endif
