#ifndef portable_bddc_preconditioner_h
#define portable_bddc_preconditioner_h

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/subdomain_dof_handler.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  enum class BDDCVariant
  {
    corner,
    corner_edge,
    corner_edge_face
  };

  template <int dim, typename Number>
  class BDDCPreconditioner
  {
  public:
    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    BDDCPreconditioner(const SchurInterfaceOperator<dim, Number>       &interface_operator,
                       const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
                       const BDDCVariant variant = BDDCVariant::corner_edge_face);

    // void
    // vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //       const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

    void
    project_to_homogeneous_constraints(InterfaceVectorType &dst) const;

    // void
    // project(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //         const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

    // void
    // balance(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //         const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

    // void
    // balance_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //               const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    //               const bool computation_on,
    //               const bool communication_on) const;

    // void
    // balance_and_vmult(
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &S_per_dst,
    //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

    // void
    // setup_coarse_matrix();

    void
    coarse_to_global_interface(InterfaceVectorType  &interface_vector,
                               const Vector<Number> &coarse_vector) const;

    // void
    // coarse_to_global_interface_and_S_update(
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &interface_vector,
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &S_per_interface_vector,
    //   const Vector<Number>                                             &coarse_vector) const;

    // void
    // vmult_enhanced(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &z,
    //                LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &s_tilde,
    //                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &r)
    //                const;

    void
    global_interface_to_coarse(Vector<Number>            &coarse_vector,
                               const InterfaceVectorType &interface_vector) const;

    void
    reset_timings() const;


    const std::array<double, 4> &
    get_timings() const;

    // void
    // vmult_interface(
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

  private:
    void
    setup_primal_constraint_views();

    ObserverPointer<const SchurInterfaceOperator<dim, Number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    LAPACKFullMatrix<Number> coarse_matrix;

    BDDCVariant bddc_variant;

    unsigned int n_global_coarse_dofs;
    unsigned int n_local_coarse_dofs;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const InterfaceVectorType &interface_weights;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    // Flattened fine local interface DoF indices associated with each local primal constraint
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_dofs;

    // offset of the primal constraint dofs per each dof entity
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_offsets;

    // subdomain (local) to global constraint map
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      subdomain_to_global_coarse_dofs;

    std::vector<unsigned int> subdomain_to_global_coarse_dofs_vector_host;

    mutable InterfaceVectorType temp_interface;

    //  mutable InterfaceVectorType z0, S_z0;

    // std::vector<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>
    //   S_per_coarse_basis_functions;

    mutable std::vector<Number> temp_coarse_gather;
    mutable std::vector<Number> temp_local_coarse;

    // mutable Vector<Number> temp_coarse_rhs;
    // mutable Vector<Number> temp_coarse_solution;

    /**
     * timings[0] = Dirichler solve
     * timings[1] = Neumann solve
     * timings[2] = coarse solve
     * timings[3] = projection step
     */
    mutable std::array<double, 4> timings;
  };

  template <int dim, typename Number>
  BDDCPreconditioner<dim, Number>::BDDCPreconditioner(
    const SchurInterfaceOperator<dim, Number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
    const BDDCVariant                                variant)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , coarse_problem_rank(subdomain_dof_handler->n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
  {
    if (dim == 2 && variant == BDDCVariant::corner_edge_face)
      bddc_variant = BDDCVariant::corner_edge;
    else
      bddc_variant = variant;

    // store primal constraints
    {
      const auto &subdomain_dof_info = this->subdomain_dof_handler->get_dof_info();

      if (bddc_variant == BDDCVariant::corner)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[1]; // End of Vertices
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[1];
        }
      else if (bddc_variant == BDDCVariant::corner_edge)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[2]; // End of Edges
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[2];
        }
      else // corner_edge_face
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[3]; // End of Faces
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[3];
        }

      setup_primal_constraint_views();
    }

    temp_interface.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    // S_per_coarse_basis_functions.resize(n_subdomains);

    // for (unsigned int i = 0; i < n_subdomains; ++i)
    //   S_per_coarse_basis_functions[i].reinit(temp_interface);

    // z0.reinit(temp_interface);
    // S_z0.reinit(temp_interface);

    temp_coarse_gather.resize(n_global_coarse_dofs);
    temp_local_coarse.resize(n_local_coarse_dofs);

    // temp_coarse_rhs.reinit(n_global_coarse_dofs);
    // temp_coarse_solution.reinit(n_global_coarse_dofs);
  }

  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::reset_timings() const
  {
    for (unsigned int i = 0; i < timings.size(); ++i)
      timings[i] = 0.;
  }

  template <int dim, typename Number>
  const std::array<double, 4> &
  BDDCPreconditioner<dim, Number>::get_timings() const
  {
    return timings;
  }

  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::setup_primal_constraint_views()
  {
    const auto &dof_info          = this->subdomain_dof_handler->get_dof_info();
    const auto &local_constraints = dof_info.local_primal_constraints;

    std::vector<unsigned int> constraint_dofs_v;
    std::vector<unsigned int> constraint_dofs_offsets_v;

    subdomain_to_global_coarse_dofs_vector_host.clear();

    std::vector<Number> coarse_weights_v;

    constraint_dofs_offsets_v.push_back(0);
    for (const auto &constraint : local_constraints)
      {
        if (bddc_variant == BDDCVariant::corner && constraint.type != PrimalConstraintType::Vertex)
          continue;

        if (bddc_variant == BDDCVariant::corner_edge &&
            constraint.type == PrimalConstraintType::Face)
          continue;

        constraint_dofs_v.insert(constraint_dofs_v.end(),
                                 constraint.local_subdomain_dofs.begin(),
                                 constraint.local_subdomain_dofs.end());
        constraint_dofs_offsets_v.push_back(constraint_dofs_v.size());
        subdomain_to_global_coarse_dofs_vector_host.push_back(constraint.global_coarse_dof_index);

        coarse_weights_v.push_back(1.0 / Number(constraint.local_subdomain_dofs.size()));
      }

    // Allocate and copy to Device Kokkos::Views
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_dofs_host"),
      constraint_dofs_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_constraint_offsets_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_constraint_offsets_host"),
      constraint_dofs_offsets_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> subdomain_to_global_coarse_dofs_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subdomain_to_global_primal_constraint_dofs_host"),
      subdomain_to_global_coarse_dofs_vector_host.size());


    Kokkos::View<Number *, Kokkos::HostSpace> coarse_weights_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_weights_host"),
      coarse_weights_v.size());

    std::copy(constraint_dofs_v.begin(),
              constraint_dofs_v.end(),
              primal_constraint_dofs_host.data());
    std::copy(constraint_dofs_offsets_v.begin(),
              constraint_dofs_offsets_v.end(),
              primal_constraint_constraint_offsets_host.data());
    std::copy(subdomain_to_global_coarse_dofs_vector_host.begin(),
              subdomain_to_global_coarse_dofs_vector_host.end(),
              subdomain_to_global_coarse_dofs_host.data());
    std::copy(coarse_weights_v.begin(), coarse_weights_v.end(), coarse_weights_host.data());

    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

    this->primal_constraint_dofs =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_host);
    this->primal_constraint_offsets =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_constraint_offsets_host);
    this->subdomain_to_global_coarse_dofs =
      Kokkos::create_mirror_view_and_copy(exec_space, subdomain_to_global_coarse_dofs_host);
    this->coarse_weights = Kokkos::create_mirror_view_and_copy(exec_space, coarse_weights_host);

    exec_space.fence();
  }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::project_to_homogeneous_constraints(
    InterfaceVectorType &dst) const
  {
    const auto dst_view = dst.get_values();

    const auto offsets                   = this->primal_constraint_offsets;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs;

    const unsigned int n_total_constrained_dofs = subdomain_constraint_dofs.extent(0);

    if (n_total_constrained_dofs > 0)
      {
        Kokkos::parallel_for(
          "project_to_homogeneous_constraints",
          n_total_constrained_dofs,
          KOKKOS_LAMBDA(const int i) {
            const unsigned int global_constraint_dof = subdomain_constraint_dofs(i);

            dst_view(global_constraint_dof) = Number(0);
          });
      }
  }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::vmult_enhanced(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &z,
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &s_tilde,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &r) const
  // {
  //   Assert(z.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //            interface vector partitioner."));
  //   Assert(r.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //           interface vector partitioner."));


  //   // local NN preconditioner
  //   this->vmult(z, r);

  //   this->vmult_interface(s_tilde, z);

  //   temp_interface = r;
  //   temp_interface -= s_tilde;

  //   Kokkos::fence();
  //   Timer time;

  //   this->balance_and_vmult(z0, S_z0, temp_interface);

  //   Kokkos::fence();
  //   timings[2] += time.wall_time();

  //   s_tilde += S_z0;
  //   z += z0;
  // }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::vmult(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  // {
  //   Assert(
  //     dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //     ExcMessage(
  //       "This function expects a vector initialized by SubdomainDoFHandler's interface vector
  //       partitioner."));
  //   Assert(
  //     src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //     ExcMessage(
  //       "This function expects a vector initialized by SubdomainDoFHandler's interface vector
  //       partitioner."));

  //   Kokkos::fence();
  //   Timer time;
  //   this->interface_operator->neumann_solve_subdomain(dst, src);

  //   Kokkos::fence();
  //   timings[1] += time.wall_time();
  // }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::vmult_interface(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  // {
  //   Assert(
  //     dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //     ExcMessage(
  //       "This function expects a vector initialized by SubdomainDoFHandler's interface vector
  //       partitioner."));
  //   Assert(
  //     src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //     ExcMessage(
  //       "This function expects a vector initialized by SubdomainDoFHandler's interface vector
  //       partitioner."));

  //   Kokkos::fence();
  //   Timer time;
  //   this->interface_operator->vmult(dst, src);

  //   Kokkos::fence();
  //   timings[0] += time.wall_time();
  // }

  /**
   * Schwarz projection step Id-(R_0^T*S_0^{-1}*R_0)*S :
   *    1. Apply interface operator S (maps values into flux)
   *    2. balance (i.e, apply R_0^T*S_0^{-1}*R_0) to retrive mean value
   *    3. Id-(R_0^T*S_0^{-1}*R_0)*S - gives compatible vector for subdomain
   * Neumann solve Result: retrieved values in the kernel space (i.e., mean
   * values) (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e.
   * compatible for the subdomain Neumann solve
   */
  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::project(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  // {
  //   Assert(dst.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //            interface vector partitioner."));
  //   Assert(src.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //           interface vector partitioner."));

  //   temp_interface = 0.;

  //   Kokkos::fence();
  //   Timer time;
  //   // dst = S*tmp
  //   this->interface_operator->vmult(temp_interface, src);

  //   Kokkos::fence();
  //   const double time_dirichlet = time.wall_time();
  //   timings[0] += time_dirichlet;

  //   time.restart();
  //   // tmp = R0^T*S_0^{-1}*R0*dst
  //   this->balance(dst, temp_interface);

  //   Kokkos::fence();
  //   timings[2] += time.wall_time();

  //   dst.sadd(-1., src);

  //   Kokkos::fence();
  //   timings[3] += time.wall_time() + time_dirichlet;
  // }

  /**
   * Balancing step R_0^T*S_0^{-1}*R_0:
   *    1. project from global interface to coarse space
   *    2. solve coarse problem
   *    3. project back to the global interface
   * Result: retrieved values in the kernel space (i.e., mean values)
   * (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e. compatible for
   * the subdomain Neumann solve
   */
  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::balance(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  // {
  //   Assert(dst.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //            interface vector partitioner."));
  //   Assert(src.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //           interface vector partitioner."));

  //   this->temp_coarse_rhs = 0.;

  //   // project from global interface to coarse space
  //   this->global_interface_to_coarse(this->temp_coarse_rhs, src);

  //   // solve coarse problem
  //   if (this->this_subdomain == this->coarse_problem_rank)
  //     {
  //       this->temp_coarse_solution = 0.;

  //       this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
  //     }

  //   // project back to the global interface space
  //   this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  // }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::balance_and_vmult(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &S_per_dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  // {
  //   Assert(dst.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //            interface vector partitioner."));
  //   Assert(src.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //           interface vector partitioner."));

  //   this->temp_coarse_rhs = 0.;

  //   // project from global interface to coarse space
  //   this->global_interface_to_coarse(this->temp_coarse_rhs, src);

  //   // solve coarse problem
  //   if (this->this_subdomain == this->coarse_problem_rank)
  //     {
  //       this->temp_coarse_solution = 0.;

  //       this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
  //     }

  //   // project back to the global interface space
  //   this->coarse_to_global_interface_and_S_update(dst, S_per_dst, this->temp_coarse_solution);
  // }


  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::balance_dummy(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
  //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
  //   const bool                                                              computation_on,
  //   const bool                                                              communication_on)
  //   const
  // {
  //   Assert(dst.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //            interface vector partitioner."));
  //   Assert(src.get_partitioner() ==
  //   this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's
  //           interface vector partitioner."));

  //   this->temp_coarse_rhs = 0.;

  //   // project from global interface to coarse space
  //   if (communication_on)
  //     this->global_interface_to_coarse(this->temp_coarse_rhs, src);

  //   if (computation_on)
  //     {
  //       // solve coarse problem
  //       if (this->this_subdomain == this->coarse_problem_rank)
  //         {
  //           this->temp_coarse_solution = 0.;

  //           this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
  //         }
  //     }

  //   // project back to the global interface space
  //   if (communication_on)
  //     this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  // }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::global_interface_to_coarse(
    Vector<Number>            &coarse_vector,
    const InterfaceVectorType &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);
    coarse_vector = 0;

    interface_vector.update_ghost_values();

    auto interface_vector_view = interface_vector.get_values();

    const auto weights = this->coarse_weights;

    const auto offsets                  = this->primal_constraint_offsets;
    const auto subdomain_constaint_dofs = this->primal_constraint_dofs;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> local_coarse_contribution(
      "local_coarse_contribution", this->n_local_coarse_dofs);

    Kokkos::parallel_for(
      "global_interface_to_coarse_local_sum",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        Number sum = 0;
        for (int i = start; i < end; ++i)
          {
            const unsigned int global_constraint_dof = subdomain_constaint_dofs(i);
            sum += interface_vector_view(global_constraint_dof);
          }

        sum *= weights(coarse_local_idx);

        local_coarse_contribution(coarse_local_idx) = sum;
      });

    auto local_coarse_contribution_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_coarse_contribution);

    std::fill(temp_coarse_gather.begin(), temp_coarse_gather.end(), Number(0));

    for (unsigned int i = 0; i < this->n_local_coarse_dofs; ++i)
      {
        const unsigned int global_coarse_idx  = subdomain_to_global_coarse_dofs_vector_host[i];
        temp_coarse_gather[global_coarse_idx] = local_coarse_contribution_host(i);
      }

    Utilities::MPI::sum(temp_coarse_gather,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        coarse_vector.get_values());

    interface_vector.zero_out_ghost_values();
  }

  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::coarse_to_global_interface(
    InterfaceVectorType  &interface_vector,
    const Vector<Number> &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);
    interface_vector = 0;

    std::fill(temp_local_coarse.begin(), temp_local_coarse.end(), Number(0));

    for (unsigned int i = 0; i < this->n_local_coarse_dofs; ++i)
      {
        const unsigned int global_coarse_idx = subdomain_to_global_coarse_dofs_vector_host[i];

        temp_local_coarse[i] = coarse_vector(global_coarse_idx);
      }

    // copy to host
    Kokkos::View<Number *, Kokkos::HostSpace> local_coarse_host_view(temp_local_coarse.data(),
                                                                     this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);

    auto       interface_vector_view = interface_vector.get_values();
    const auto weights               = this->coarse_weights;

    const auto offsets                  = this->primal_constraint_offsets;
    const auto subdomain_constaint_dofs = this->primal_constraint_dofs;

    Kokkos::parallel_for(
      "coarse_to_global_interface_interpolate",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const Number coarse_value =
          weights(coarse_local_idx) * local_coarse_device_view(coarse_local_idx);

        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        for (unsigned int i = start; i < end; ++i)
          {
            const unsigned int global_constraint_dof = subdomain_constaint_dofs(i);

            interface_vector_view(global_constraint_dof) = coarse_value;
          }
      });

    // update globally
    interface_vector.compress(VectorOperation::insert);
  }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::coarse_to_global_interface_and_S_update(
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &interface_vector,
  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &S_per_interface_vector,
  //   const Vector<Number>                                             &coarse_vector) const
  // {
  //   Assert(interface_vector.get_partitioner() ==
  //            this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //          ExcMessage("Interface vector is not initialized correctly."));

  //   DeviceVector<Number> interface_vector_view(interface_vector.get_values(),
  //                                              interface_vector.size());


  //   DeviceVector<Number> weights_view(interface_weights.get_values(), interface_weights.size());

  //   // copy coarse Vector to std::vector for MPi::scatter
  //   if (this->this_subdomain == this->coarse_problem_rank)
  //     {
  //       for (unsigned int i = 0; i < coarse_vector.size(); ++i)
  //         this->temp_coarse_gather[i] = coarse_vector[i];
  //     }

  //   // retrieve subdomain coarse value (i.e., mean value)
  //   // const Number subdomain_coarse_value = Utilities::MPI::scatter(
  //   //   this->subdomain_dof_handler->get_mpi_communicator(),
  //   //   this->temp_coarse_gather,
  //   //   this->coarse_problem_rank);

  //   temp_coarse_broadcast =
  //     Utilities::MPI::broadcast(this->subdomain_dof_handler->get_mpi_communicator(),
  //                               this->temp_coarse_gather,
  //                               this->coarse_problem_rank);

  //   const Number subdomain_coarse_value = temp_coarse_broadcast[this_subdomain];

  //   // propagate coarse value to the interface by applying weights
  //   interface_vector = 0.;
  //   Kokkos::parallel_for(
  //     "SubdomainLaplaceOperator::coarse_to_subdomain_interface",
  //     interface_dof_indices_subdomain.size(),
  //     KOKKOS_LAMBDA(const unsigned int i) {
  //       interface_vector_view(i) = subdomain_coarse_value * weights_view(i);
  //     });

  //   // condense
  //   interface_vector.compress(VectorOperation::add);

  //   S_per_interface_vector = 0;
  //   for (unsigned int i = 0; i < n_subdomains; ++i)
  //     S_per_interface_vector.add(temp_coarse_broadcast[i], S_per_coarse_basis_functions[i]);

  //   S_per_interface_vector.compress(VectorOperation::add);
  // }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::setup_coarse_matrix()
  // {
  //   if (this->this_subdomain == this->coarse_problem_rank)
  //     coarse_matrix.reinit(this->n_subdomains, this->n_subdomains);

  //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> phi_j(
  //     this->subdomain_dof_handler->get_interface_vector_partitioner()),
  //     S_phi_j(this->subdomain_dof_handler->get_interface_vector_partitioner());

  //   Vector<Number> e_j(this->n_subdomains), coarse_column(this->n_subdomains);

  //   for (unsigned int j = 0; j < this->n_subdomains; ++j)
  //     {
  //       e_j = 0.;

  //       if (this->this_subdomain == this->coarse_problem_rank)
  //         e_j[j] = 1.;

  //       this->coarse_to_global_interface(phi_j, e_j);

  //       this->interface_operator->vmult(S_per_coarse_basis_functions[j], phi_j);

  //       this->global_interface_to_coarse(coarse_column, S_per_coarse_basis_functions[j]);

  //       if (this->this_subdomain == this->coarse_problem_rank)
  //         for (unsigned int i = 0; i < this->n_subdomains; ++i)
  //           coarse_matrix(i, j) = coarse_column[i];
  //     }

  //   if (this->this_subdomain == this->coarse_problem_rank)
  //     {
  //       coarse_matrix.compute_inverse_svd(1e-12);

  //       // std::cout << "Singular values of the coarse matrix: " << std::endl;
  //       // for (unsigned int i = 0; i < this->n_subdomains; ++i)
  //       //   std::cout << coarse_matrix.singular_value(i) << " ";
  //       // std::cout << std::endl;
  //     }
  // }

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif