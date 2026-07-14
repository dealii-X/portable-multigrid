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

enum class PrimalConstraintType
{
  Vertex,
  Edge,
  Face
};

struct LocalPrimalConstraint
{
  // Primal constraint type
  PrimalConstraintType type;

  // The subdomains that share this specific entity.
  std::set<unsigned int> sharing_subdomains;

  // The local subdomain DoF indices that belong to this specific constraint.
  std::vector<unsigned int> local_subdomain_dofs;

  // The global DoF indices (from the distributed_dof_handler) for these DoFs.
  std::vector<types::global_dof_index> global_dof_indices;

  /**
   * The unique global index assigned to this constraint in the global coarse space S_C.
   */
  unsigned int global_coarse_dof_index;
};

struct GlobalPrimalConstraint
{
  // Primal constraint type
  PrimalConstraintType type;

  // The subdomains that share this specific entity.
  std::set<unsigned int> sharing_subdomains;

  // The global DoF indices (from the distributed_dof_handler) for these DoFs.
  std::vector<types::global_dof_index> global_dof_indices;

  /**
   * The unique global index assigned to this constraint in the global coarse space S_C.
   */
  unsigned int global_coarse_dof_index;
};



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
   * Global interface primal constraints.
   */

  std::vector<GlobalPrimalConstraint> global_primal_constraints;

  /**
   * Local offsets within my_coarse_dof_indices:
   * local_coarse_offsets[0] = Start of global Vertices (0)
   * local_coarse_offsets[1] = Start of global Edges (End of Vertices)
   * local_coarse_offsets[2] = Start of global Faces (End of Edges)
   * local_coarse_offsets[3] = End of global Faces (Total size)
   */
  std::array<unsigned int, 4> global_coarse_offsets;

  /**
   * Local (subdomain) interface primal constraints.
   */
  std::vector<LocalPrimalConstraint> local_primal_constraints;

  /**
   * Local offsets within subdomain_coarse_dof_indices:
   * local_coarse_offsets[0] = Start of local Vertices (0)
   * local_coarse_offsets[1] = Start of local Edges (End of Vertices)
   * local_coarse_offsets[2] = Start of local Faces (End of Edges)
   * local_coarse_offsets[3] = End of local Faces (Total size)
   */
  std::array<unsigned int, 4> local_coarse_offsets;

  /**
   * The subset of global coarse DoF indices that this specific subdomain participates in.
   * Sorted internally by Vertex indices, then Edge indices, then Face indices.
   */
  std::vector<unsigned int> subdomain_coarse_dof_indices;


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
    local_primal_constraints.clear();
    global_primal_constraints.clear();
    subdomain_coarse_dof_indices.clear();
    global_coarse_offsets.fill(numbers::invalid_unsigned_int);
    local_coarse_offsets.fill(numbers::invalid_unsigned_int);
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
  const IndexSet    &all_interface_dofs      = subdomain_dof_info.all_interface_dofs_global;
  const unsigned int n_global_interface_dofs = all_interface_dofs.n_elements();

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

  std::vector<GlobalPrimalConstraint> global_corner_blocks;
  std::vector<GlobalPrimalConstraint> global_edge_blocks;
  std::vector<GlobalPrimalConstraint> global_face_blocks;

  std::vector<LocalPrimalConstraint> local_corner_blocks;
  std::vector<LocalPrimalConstraint> local_edge_blocks;
  std::vector<LocalPrimalConstraint> local_face_blocks;

  // Track the original geometric class_idx for each local item to map them later
  std::vector<unsigned int> local_corner_class_idxs;
  std::vector<unsigned int> local_edge_class_idxs;
  std::vector<unsigned int> local_face_class_idxs;

  // categorize inteface dofs
  {
    IndexSet used_dofs(this->distributed_dof_handler->n_dofs());

    for (unsigned int class_idx = 0; class_idx < equivalent_dof_classes.size(); ++class_idx)
      {
        const auto &class_set     = equivalent_dof_classes[class_idx];
        const auto &dofs_in_class = dofs_per_equivalence_class[class_idx];

        bool is_valid_constraint = false;

        PrimalConstraintType primal_constraint_type;

        if (dofs_in_class.size() == 1 && class_set.size() > 2)
          {
            // vertex (corner) interface dofs are shared by a unique maximal set of processors (more
            // than 2)
            primal_constraint_type = PrimalConstraintType::Vertex;

            is_valid_constraint = true;

            for (const auto global_idx : dofs_in_class)
              used_dofs.add_index(global_idx);
          }
        else if constexpr (dim == 2)
          {
            // in 2d all other interface dofs are edge dofs
            primal_constraint_type = PrimalConstraintType::Edge;

            is_valid_constraint = true;

            for (const auto global_idx : dofs_in_class)
              used_dofs.add_index(global_idx);
          }

        else if constexpr (dim == 3)
          {
            if ((class_set.size() == 2) && (dofs_in_class.size() > 1))
              {
                // face dofs in 3d are shared by exactly two subdomains
                // we also filter out case there's only one dof shared by two subdomains

                primal_constraint_type = PrimalConstraintType::Face;

                is_valid_constraint = true;

                for (const auto global_idx : dofs_in_class)
                  used_dofs.add_index(global_idx);
              }
            else
              {
                // all other interface dofs in 3d are edge dofs

                primal_constraint_type = PrimalConstraintType::Edge;

                is_valid_constraint = true;

                for (const auto global_idx : dofs_in_class)
                  used_dofs.add_index(global_idx);
              }
          }

        if (is_valid_constraint)
          {
            GlobalPrimalConstraint global_primal_constraint;
            global_primal_constraint.type               = primal_constraint_type;
            global_primal_constraint.sharing_subdomains = class_set;

            // assign temporarily, will be reassigned right after this loop
            global_primal_constraint.global_coarse_dof_index = class_idx;

            for (const auto global_idx : dofs_in_class)
              global_primal_constraint.global_dof_indices.push_back(global_idx);

            if (global_primal_constraint.type == PrimalConstraintType::Vertex)
              {
                global_corner_blocks.push_back(global_primal_constraint);
              }
            else if (global_primal_constraint.type == PrimalConstraintType::Edge)
              {
                global_edge_blocks.push_back(global_primal_constraint);
              }
            else if (global_primal_constraint.type == PrimalConstraintType::Face)
              {
                global_face_blocks.push_back(global_primal_constraint);
              }

            // if this subdomain partecipates in this class
            if (class_set.count(this->subdomain_id) > 0)
              {
                LocalPrimalConstraint local_primal_constraint;
                local_primal_constraint.type               = primal_constraint_type;
                local_primal_constraint.sharing_subdomains = class_set;

                // assign temporarily, will be reassigned right after this loop
                local_primal_constraint.global_coarse_dof_index = class_idx;

                for (const auto global_idx : dofs_in_class)
                  {
                    // Only add DoFs that actually belong to our subdomain's interface
                    if (subdomain_dof_info.subdomain_interface_dofs_global.is_element(global_idx))
                      {
                        local_primal_constraint.global_dof_indices.push_back(global_idx);
                        local_primal_constraint.local_subdomain_dofs.push_back(
                          subdomain_dof_info.global_to_subdomain_interface_map.at(global_idx));
                      }
                  }

                if (local_primal_constraint.type == PrimalConstraintType::Vertex)
                  {
                    local_corner_blocks.push_back(local_primal_constraint);
                    local_corner_class_idxs.push_back(class_idx);
                  }
                else if (local_primal_constraint.type == PrimalConstraintType::Edge)
                  {
                    local_edge_blocks.push_back(local_primal_constraint);
                    local_edge_class_idxs.push_back(class_idx);
                  }
                else if (local_primal_constraint.type == PrimalConstraintType::Face)
                  {
                    local_face_blocks.push_back(local_primal_constraint);
                    local_face_class_idxs.push_back(class_idx);
                  }
              }
          }
      }

    used_dofs.compress();
    AssertThrow(
      all_interface_dofs == used_dofs,
      ExcMessage(
        "SubdomainDoFHandler<dim>::categorize_interface_dofs() couldn't categorize some of the dofs."));
  }

  // compute global coarse dofs offsets
  {
    subdomain_dof_info.global_coarse_offsets[0] = 0;
    subdomain_dof_info.global_coarse_offsets[1] = global_corner_blocks.size();
    subdomain_dof_info.global_coarse_offsets[2] =
      global_corner_blocks.size() + global_edge_blocks.size();
    subdomain_dof_info.global_coarse_offsets[3] =
      global_corner_blocks.size() + global_edge_blocks.size() + global_face_blocks.size();
  }

  // Create a mapping from original geometric class_idx to the new, sequential global coarse index
  std::map<unsigned int, unsigned int> class_idx_to_global_sequential;

  // enumerate global coarse dofs
  {
    // 1. vertices (corners)
    for (unsigned int i = 0; i < global_corner_blocks.size(); ++i)
      {
        const unsigned int old_idx = global_corner_blocks[i].global_coarse_dof_index;

        const unsigned int new_idx = subdomain_dof_info.global_coarse_offsets[0] + i;

        global_corner_blocks[i].global_coarse_dof_index = new_idx;

        class_idx_to_global_sequential[old_idx] = new_idx;
      }

    // 2. edges
    for (unsigned int i = 0; i < global_edge_blocks.size(); ++i)
      {
        const unsigned int old_idx = global_edge_blocks[i].global_coarse_dof_index;

        const unsigned int new_idx = subdomain_dof_info.global_coarse_offsets[1] + i;

        global_edge_blocks[i].global_coarse_dof_index = new_idx;

        class_idx_to_global_sequential[old_idx] = new_idx;
      }

    // 3. faces
    for (unsigned int i = 0; i < global_face_blocks.size(); ++i)
      {
        const unsigned int old_idx = global_face_blocks[i].global_coarse_dof_index;

        const unsigned int new_idx = subdomain_dof_info.global_coarse_offsets[2] + i;

        global_face_blocks[i].global_coarse_dof_index = new_idx;

        class_idx_to_global_sequential[old_idx] = new_idx;
      }
  }

  // store global coarse dofs in the new numbering
  {
    subdomain_dof_info.global_primal_constraints.clear();

    subdomain_dof_info.global_primal_constraints.insert(
      subdomain_dof_info.global_primal_constraints.end(),
      global_corner_blocks.begin(),
      global_corner_blocks.end());

    subdomain_dof_info.global_primal_constraints.insert(
      subdomain_dof_info.global_primal_constraints.end(),
      global_edge_blocks.begin(),
      global_edge_blocks.end());
    subdomain_dof_info.global_primal_constraints.insert(
      subdomain_dof_info.global_primal_constraints.end(),
      global_face_blocks.begin(),
      global_face_blocks.end());
  }

  // Map the local subdomain coarse dof to the new global numbering
  {
    for (unsigned int i = 0; i < local_corner_blocks.size(); ++i)
      {
        const unsigned int old_idx                     = local_corner_class_idxs[i];
        local_corner_blocks[i].global_coarse_dof_index = class_idx_to_global_sequential.at(old_idx);
      }

    for (unsigned int i = 0; i < local_edge_blocks.size(); ++i)
      {
        const unsigned int old_idx                   = local_edge_class_idxs[i];
        local_edge_blocks[i].global_coarse_dof_index = class_idx_to_global_sequential.at(old_idx);
      }

    for (unsigned int i = 0; i < local_face_blocks.size(); ++i)
      {
        const unsigned int old_idx                   = local_face_class_idxs[i];
        local_face_blocks[i].global_coarse_dof_index = class_idx_to_global_sequential.at(old_idx);
      }
  }

  // compute local offsets for each coarse dof enity and store in  flattened format
  {
    subdomain_dof_info.local_coarse_offsets[0] = 0;
    subdomain_dof_info.local_coarse_offsets[1] = local_corner_blocks.size();
    subdomain_dof_info.local_coarse_offsets[2] =
      local_corner_blocks.size() + local_edge_blocks.size();
    subdomain_dof_info.local_coarse_offsets[3] =
      local_corner_blocks.size() + local_edge_blocks.size() + local_face_blocks.size();

    subdomain_dof_info.local_primal_constraints.clear();

    subdomain_dof_info.local_primal_constraints.insert(
      subdomain_dof_info.local_primal_constraints.end(),
      local_corner_blocks.begin(),
      local_corner_blocks.end());

    subdomain_dof_info.local_primal_constraints.insert(
      subdomain_dof_info.local_primal_constraints.end(),
      local_edge_blocks.begin(),
      local_edge_blocks.end());

    subdomain_dof_info.local_primal_constraints.insert(
      subdomain_dof_info.local_primal_constraints.end(),
      local_face_blocks.begin(),
      local_face_blocks.end());
  }

  // fill local (subdomain) coarse dof indices
  {
    subdomain_dof_info.subdomain_coarse_dof_indices.clear();

    for (const auto &block : local_corner_blocks)
      subdomain_dof_info.subdomain_coarse_dof_indices.push_back(block.global_coarse_dof_index);

    for (const auto &block : local_edge_blocks)
      subdomain_dof_info.subdomain_coarse_dof_indices.push_back(block.global_coarse_dof_index);

    for (const auto &block : local_face_blocks)
      subdomain_dof_info.subdomain_coarse_dof_indices.push_back(block.global_coarse_dof_index);
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
