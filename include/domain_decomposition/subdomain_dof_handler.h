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
   * Subdomain interface vertex (corner) DoFs.
   */
  IndexSet interface_vertex_dofs;

  /**
   * Subdomain interface edge DoFs.
   */
  IndexSet interface_edge_dofs;

  /**
   * Subdomain interface face DoFs in 3d.
   */
  IndexSet interface_face_dofs;


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
    interface_vertex_dofs.clear();
    interface_vertex_dofs.clear();
    interface_vertex_dofs.clear();
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
  categorize_interface_dofs(const std::vector<IndexSet> &all_interface_dof_sets,
                            const IndexSet              &all_inteface_dofs);

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

  IndexSet &local_interface_indices = subdomain_dof_info.subdomain_interface_dofs_global;

  std::vector<IndexSet> all_sets =
    Utilities::MPI::all_gather(this->get_mpi_communicator(), local_interface_indices);

  IndexSet all_interface_dofs(local_interface_indices.size());
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

  this->categorize_interface_dofs(all_sets, all_interface_dofs);
}


template <int dim>
void
SubdomainDoFHandler<dim>::categorize_interface_dofs(
  const std::vector<IndexSet> &all_interface_dof_sets,
  const IndexSet              &all_inteface_dofs)
{
  subdomain_dof_info.interface_vertex_dofs.set_size(this->distributed_dof_handler->n_dofs());
  subdomain_dof_info.interface_edge_dofs.set_size(this->distributed_dof_handler->n_dofs());
  subdomain_dof_info.interface_face_dofs.set_size(this->distributed_dof_handler->n_dofs());

  const unsigned int n_global_interface_dofs = all_inteface_dofs.n_elements();

  std::map<std::set<unsigned int>, std::vector<std::pair<types::global_dof_index, unsigned int>>>
    equivalence_dof_classes;

  std::vector<std::set<unsigned int>> subdomains_per_dof(n_global_interface_dofs);
  std::vector<std::pair<types::global_dof_index, unsigned int>> dof_pairs(n_global_interface_dofs);


  for (unsigned int i = 0; i < n_global_interface_dofs; ++i)
    {
      const auto global_idx = all_inteface_dofs.nth_index_in_set(i);
      // const unsigned int local_subdomain_idx =
      //   subdomain_dof_info.global_to_subdomain_interface_map.at(global_idx);

      dof_pairs[i] = std::make_pair(global_idx, 1);

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

  std::vector<std::vector<unsigned int>> subdomain_dofs_per_class(equivalent_dof_classes.size());

  for (unsigned int i = 0; i < n_global_interface_dofs; ++i)
    {
      auto it = std::lower_bound(equivalent_dof_classes.begin(),
                                 equivalent_dof_classes.end(),
                                 subdomains_per_dof[i]);

      unsigned int class_idx = std::distance(equivalent_dof_classes.begin(), it);

      // if (subdomain_dof_info.subdomain_interface_dofs.is_element(dof_pairs[i].second))
      subdomain_dofs_per_class[class_idx].push_back(dof_pairs[i].first);
    }


  for (unsigned int p = 0; p < this->n_subdomains(); ++p)
    {
      if (p == this->get_subdomain_id())
        {
          MPI_Barrier(this->get_mpi_communicator());
          std::cout << "On subdomain " << p << ", n_classes =  " << equivalent_dof_classes.size()
                    << ": " << std::endl;
          for (unsigned int class_idx = 0; class_idx < equivalent_dof_classes.size(); ++class_idx)
            {
              std::cout << "Class " << class_idx << " ("
                        << subdomain_dofs_per_class[class_idx].size() << ")"
                        << ":  ";
              // for (unsigned int i = 0; i < equivalent_dof_classes[class_idx].size(); ++i)
              for (const auto &key : equivalent_dof_classes[class_idx])
                std::cout << key << "    ";
              std::cout << std::endl;
              for (const auto &key : subdomain_dofs_per_class[class_idx])
                std::cout << key << "    ";
              std::cout << std::endl;
            }
          std::cout << std::endl;

          MPI_Barrier(this->get_mpi_communicator());
        }
      std::cout << std::endl;
    }

  // categorize inteface dofs
  for (unsigned int class_idx = 0; class_idx < equivalent_dof_classes.size(); ++class_idx)
    {
      const auto &class_set     = equivalent_dof_classes[class_idx];
      const auto &dofs_in_class = subdomain_dofs_per_class[class_idx];


      if (class_set.size() == 2)
        {
          // face dofs in 3d and edge dofs in 2d are shared by exactly two subdomains

          if constexpr (dim == 2)
            {
              // subdomain_dof_info.interface_edge_dofs.insert(
              //   subdomain_dof_info.interface_edge_dofs.end(),
              //   dofs_in_class.begin(),
              //   dofs_in_class.end());
              for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
                subdomain_dof_info.interface_edge_dofs.add_index(dofs_in_class[i]);
            }

          if constexpr (dim == 3)
            {
              // subdomain_dof_info.interface_face_dofs.insert(
              //   subdomain_dof_info.interface_face_dofs.end(),
              //   dofs_in_class.begin(),
              //   dofs_in_class.end());

              for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
                subdomain_dof_info.interface_face_dofs.add_index(dofs_in_class[i]);
            }
        }
      else if constexpr (dim == 2)
        {
          // rest of the dofs in 2d are vertex (corner) dofs

          // subdomain_dof_info.interface_vertex_dofs.insert(
          //   subdomain_dof_info.interface_vertex_dofs.end(),
          //   dofs_in_class.begin(),
          //   dofs_in_class.end());

          for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
            subdomain_dof_info.interface_vertex_dofs.add_index(dofs_in_class[i]);
        }
      else if constexpr (dim == 3)
        {
          if (dofs_in_class.size() == 1)
            {
              subdomain_dof_info.interface_vertex_dofs.add_index(dofs_in_class[0]);
            }
          else
            {
              // subdomain_dof_info.interface_edge_dofs.insert(
              //   subdomain_dof_info.interface_edge_dofs.end(),
              //   dofs_in_class.begin(),
              //   dofs_in_class.end());

              for (unsigned int i = 0; i < dofs_in_class.size(); ++i)
                subdomain_dof_info.interface_edge_dofs.add_index(dofs_in_class[i]);
            }
        }
    }



  // for (unsigned int i = 0; i < n_global_interface_dofs; ++i)

  //   for (const auto &global_idx : subdomain_dof_info.interface_dofs_global)
  //     {
  //       std::set<unsigned int> equivalence_set;

  //       const unsigned int local_subdomain_idx =
  //         subdomain_dof_info.global_to_subdomain_interface_map[global_idx];

  //       std::pair<types::global_dof_index, unsigned int> dof_ids =
  //         std::make_pair(global_idx, local_subdomain_idx);

  //       for (unsigned int p = 0; p < this->n_subdomains(); ++p)
  //         {
  //           if (all_interface_dof_sets[p].is_element(global_idx))
  //             equivalence_set.insert(p);
  //         }

  //       if (equivalence_set.size() == 2) // this dof is on edge in 2d or on face in 3d
  //         {
  //           if constexpr (dim == 3)
  //             subdomain_dof_info.interface_face_dofs.push_back(dof_ids);

  //           if constexpr (dim == 2)
  //             subdomain_dof_info.interface_edge_dofs.push_back(dof_ids);
  //         }
  //       else if constexpr (dim == 2) // in 2d all other dofs are at vertices (corners)
  //         {
  //           subdomain_dof_info.interface_vertex_dofs.push_back(dof_ids);
  //         }
  //       else // group by subdomains sharing dof to identify 3d edges vs vertices
  //         {
  //           equivalence_dof_classes[equivalence_set].push_back(dof_ids);
  //         }
  //     }

  // // in 3d a corner vertex is defined by maximal sets, and the rest are edge dofs
  // if constexpr (dim == 3)
  //   {
  //     std::vector<std::set<unsigned int>> multi_shared_sets;
  //     for (const auto &[equiv_set, dofs] : equivalence_dof_classes)
  //       {
  //         multi_shared_sets.push_back(equiv_set);
  //       }

  //     for (const auto &[equiv_set, dofs_in_class] : equivalence_dof_classes)
  //       {
  //         bool is_strict_subset = false;

  //         for (const auto &other_set : multi_shared_sets)
  //           {
  //             if (equiv_set != other_set && std::includes(other_set.begin(),
  //                                                         other_set.end(),
  //                                                         equiv_set.begin(),
  //                                                         equiv_set.end()))
  //               {
  //                 is_strict_subset = true;
  //                 break;
  //               }
  //           }

  //         if (!is_strict_subset)
  //           {
  //             subdomain_dof_info.interface_vertex_dofs.push_back(dofs_in_class[0]);

  //             for (size_t i = 1; i < dofs_in_class.size(); ++i)
  //               {
  //                 subdomain_dof_info.interface_edge_dofs.push_back(dofs_in_class[i]);
  //               }
  //           }
  //         else
  //           {
  //             for (const auto &dof : dofs_in_class)
  //               subdomain_dof_info.interface_edge_dofs.push_back(dof);
  //           }
  //       }
  //   }

  for (unsigned int p = 0; p < this->n_subdomains(); ++p)
    {
      MPI_Barrier(this->get_mpi_communicator());
      if (p == this->get_subdomain_id())
        {
          std::cout << "On subdomain " << p << ", vertex_dofs: ";
          // for (unsigned int i = 0; i < subdomain_dof_info.interface_vertex_dofs.size();
          // ++i)
          for (const auto &index : subdomain_dof_info.interface_vertex_dofs)
            std::cout << index << "    ";
          std::cout << std::endl;

          std::cout << "On subdomain " << p << ", edge dofs: ";
          // for (unsigned int i = 0; i < subdomain_dof_info.interface_edge_dofs.size(); ++i)
          //   std::cout << subdomain_dof_info.interface_edge_dofs[i].second << " / "
          //             << subdomain_dof_info.interface_edge_dofs[i].first << "    ";
          // std::cout << std::endl;

          for (const auto &index : subdomain_dof_info.interface_edge_dofs)
            std::cout << index << "    ";
          std::cout << std::endl;

          std::cout << "On subdomain " << p << ", face dofs: ";
          // for (unsigned int i = 0; i < subdomain_dof_info.interface_face_dofs.size(); ++i)
          //   std::cout << subdomain_dof_info.interface_face_dofs[i].second << " / "
          //             << subdomain_dof_info.interface_face_dofs[i].first << "    ";
          // std::cout << std::endl;
          for (const auto &index : subdomain_dof_info.interface_face_dofs)
            std::cout << index << "    ";
          std::cout << std::endl;
        }
      std::cout << std::endl;
      MPI_Barrier(this->get_mpi_communicator());
    }
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
