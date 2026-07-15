#ifndef portable_bddc_preconditioner_h
#define portable_bddc_preconditioner_h

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_projected_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"



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
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;


    BDDCPreconditioner(const SchurInterfaceOperator<dim, Number>       &interface_operator,
                       const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
                       const BDDCVariant variant = BDDCVariant::corner_edge_face);

    // void
    // vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //       const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const;

    void
    solve_subdomain_with_constraints(InterfaceVectorType       &dst,
                                     const InterfaceVectorType &src) const;

    void
    project_to_homogeneous_constraints_interface(InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_subdomain(SubdomainVectorType &subdomain_vector) const;


    void
    lift_coarse_to_interface(InterfaceVectorType  &interface_vector,
                             const Vector<Number> &coarse_vector) const;

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

    void
    setup_coarse_matrix();

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


    struct SubdomainProjectorWrapper
    {
    public:
      const BDDCPreconditioner<dim, Number> &parent;

      SubdomainProjectorWrapper(const BDDCPreconditioner<dim, Number> &parent_preconditioner)
        : parent(parent_preconditioner)
      {}

      void
      project(SubdomainVectorType &subdomain_vector) const
      {
        parent.project_to_homogeneous_constraints_subdomain(subdomain_vector);
      }
    };


    // void
    // test_subdomain_solve(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

  private:
    void
    setup_primal_constraint_views();

    void
    compute_local_coarse_matrix(LAPACKFullMatrix<Number> &local_coarse_matrix) const;

    ObserverPointer<const SchurInterfaceOperator<dim, Number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    SubdomainBDDCOperatorWrapper<dim, Number> subdomain_bddc_operator;

    LAPACKFullMatrix<Number> coarse_matrix;

    BDDCVariant bddc_variant;

    unsigned int       n_global_coarse_dofs;
    unsigned int       n_local_coarse_dofs;
    const unsigned int interface_vector_size;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const InterfaceVectorType &interface_weights;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    // Flattened fine local interface DoF indices associated with each local primal constraint
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_interface_local;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_subdomain;

    // offset of the primal constraint dofs per each dof entity
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_offsets;

    // subdomain (local) to global constraint map
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> coarse_dofs_local_to_global;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> corner_dofs_subdomain;

    std::vector<unsigned int> coarse_dofs_local_to_global_vector_host;

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

    mutable SubdomainVectorType temp_subdomain_dst;
    mutable SubdomainVectorType temp_subdomain_src;
  };

  template <int dim, typename Number>
  BDDCPreconditioner<dim, Number>::BDDCPreconditioner(
    const SchurInterfaceOperator<dim, Number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
    const BDDCVariant                                variant)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , subdomain_bddc_operator(subdomain_operator)
    , interface_vector_size(subdomain_operator.get_interface_dof_indices_subdomain().size())
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

    AssertThrow(n_global_coarse_dofs > 0, ExcMessage("There's zero global constraints"));
    AssertThrow(n_local_coarse_dofs > 0, ExcMessage("There's zero local constraints"));


    // S_per_coarse_basis_functions.resize(n_subdomains);

    // for (unsigned int i = 0; i < n_subdomains; ++i)
    //   S_per_coarse_basis_functions[i].reinit(temp_interface);

    // z0.reinit(temp_interface);
    // S_z0.reinit(temp_interface);

    temp_coarse_gather.resize(n_global_coarse_dofs);
    temp_local_coarse.resize(n_local_coarse_dofs);

    // temp_coarse_rhs.reinit(n_global_coarse_dofs);
    // temp_coarse_solution.reinit(n_global_coarse_dofs);

    subdomain_operator.initialize_dof_vector(temp_subdomain_dst);
    temp_subdomain_src.reinit(temp_subdomain_dst);
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
  BDDCPreconditioner<dim, Number>::solve_subdomain_with_constraints(
    InterfaceVectorType       &dst,
    const InterfaceVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    dst = 0;

    src.update_ghost_values();

    const auto interface_dofs_subdomain = interface_dof_indices_subdomain;

    temp_subdomain_src = 0;
    temp_subdomain_dst = 0;

    const DeviceVector<Number> src_interface_view(src.get_values(),
                                                  interface_dofs_subdomain.size());
    DeviceVector<Number>       src_subdomain_view(temp_subdomain_src.get_values(),
                                                  temp_subdomain_src.size());

    // read interface values into the subdomain vector
    Kokkos::parallel_for(
      "read_src_interface", interface_dofs_subdomain.size(), KOKKOS_LAMBDA(const int i) {
        src_subdomain_view(interface_dofs_subdomain(i)) = src_interface_view(i);
      });
    Kokkos::fence();

    SubdomainProjectorWrapper projector(*this);

    projector.project(temp_subdomain_src);

    SolverControl solver_control(1000, 1e-9 * temp_subdomain_src.l2_norm());

    SolverProjectedCG<SubdomainVectorType> solver(solver_control);

    solver.solve_projected(subdomain_bddc_operator,
                           temp_subdomain_dst,
                           temp_subdomain_src,
                           PreconditionIdentity(),
                           projector);

    projector.project(temp_subdomain_dst);

    DeviceVector<Number> dst_interface_view(dst.get_values(), interface_dofs_subdomain.size());
    DeviceVector<Number> dst_subdomain_view(temp_subdomain_dst.get_values(),
                                            temp_subdomain_dst.size());


    Kokkos::parallel_for(
      "distribute_interface_dofs", interface_dofs_subdomain.size(), KOKKOS_LAMBDA(const int i) {
        dst_interface_view(i) = dst_subdomain_view(interface_dofs_subdomain(i));
      });
    Kokkos::fence();

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    std::cout << std::endl
              << "Projected solver on subdomain " << this->subdomain_dof_handler->get_subdomain_id()
              << " converged in " << solver_control.last_step() << std::endl;
  }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::setup_primal_constraint_views()
  {
    const auto &dof_info          = this->subdomain_dof_handler->get_dof_info();
    const auto &local_constraints = dof_info.local_primal_constraints;

    std::vector<unsigned int> primal_constraint_dofs_interface_local_v;
    std::vector<unsigned int> primal_constraint_dofs_subdomain_v;
    std::vector<unsigned int> constraint_dofs_offsets_v;
    std::vector<unsigned int> corner_dofs_subdomain_v;

    coarse_dofs_local_to_global_vector_host.clear();

    std::vector<Number> coarse_weights_v;

    constraint_dofs_offsets_v.push_back(0);
    for (const auto &constraint : local_constraints)
      {
        if (bddc_variant == BDDCVariant::corner && constraint.type != PrimalConstraintType::Vertex)
          continue;

        if (bddc_variant == BDDCVariant::corner_edge &&
            constraint.type == PrimalConstraintType::Face)
          continue;

        if (constraint.type == PrimalConstraintType::Vertex)
          {
            AssertDimension(constraint.interface_partitioner_dofs_local.size(), 1);
            AssertDimension(constraint.local_subdomain_dofs.size(), 1);

            corner_dofs_subdomain_v.push_back(constraint.local_subdomain_dofs[0]);
          }

        primal_constraint_dofs_interface_local_v.insert(
          primal_constraint_dofs_interface_local_v.end(),
          constraint.interface_partitioner_dofs_local.begin(),
          constraint.interface_partitioner_dofs_local.end());

        primal_constraint_dofs_subdomain_v.insert(primal_constraint_dofs_subdomain_v.end(),
                                                  constraint.local_subdomain_dofs.begin(),
                                                  constraint.local_subdomain_dofs.end());


        constraint_dofs_offsets_v.push_back(primal_constraint_dofs_interface_local_v.size());

        coarse_dofs_local_to_global_vector_host.push_back(constraint.global_coarse_dof_index);

        Number weight = Number(1) / Number(constraint.interface_partitioner_dofs_local.size());

        AssertThrow(weight > 0, ExcInternalError());
        coarse_weights_v.push_back(weight);
      }

    AssertThrow(primal_constraint_dofs_interface_local_v.size() > 0, ExcInternalError());
    AssertThrow(primal_constraint_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(constraint_dofs_offsets_v.size() > 0, ExcInternalError());
    AssertThrow(corner_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(coarse_dofs_local_to_global_vector_host.size() > 0, ExcInternalError());
    AssertThrow(coarse_weights_v.size() > 0, ExcInternalError());


    // Allocate and copy to Device Kokkos::Views
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_interface_local_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "primal_constraint_dofs_interface_local_host"),
      primal_constraint_dofs_interface_local_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_dofs_subdomain_host"),
      primal_constraint_dofs_subdomain_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_constraint_offsets_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_constraint_offsets_host"),
      constraint_dofs_offsets_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> coarse_dofs_local_to_global_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_dofs_local_to_global_host"),
      coarse_dofs_local_to_global_vector_host.size());

    Kokkos::View<unsigned int *, Kokkos::HostSpace> corner_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "corner_dofs_subdomain_host"),
      corner_dofs_subdomain_v.size());

    Kokkos::View<Number *, Kokkos::HostSpace> coarse_weights_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_weights_host"),
      coarse_weights_v.size());

    std::copy(primal_constraint_dofs_interface_local_v.begin(),
              primal_constraint_dofs_interface_local_v.end(),
              primal_constraint_dofs_interface_local_host.data());
    std::copy(primal_constraint_dofs_subdomain_v.begin(),
              primal_constraint_dofs_subdomain_v.end(),
              primal_constraint_dofs_subdomain_host.data());
    std::copy(constraint_dofs_offsets_v.begin(),
              constraint_dofs_offsets_v.end(),
              primal_constraint_constraint_offsets_host.data());
    std::copy(coarse_dofs_local_to_global_vector_host.begin(),
              coarse_dofs_local_to_global_vector_host.end(),
              coarse_dofs_local_to_global_host.data());
    std::copy(corner_dofs_subdomain_v.begin(),
              corner_dofs_subdomain_v.end(),
              corner_dofs_subdomain_host.data());
    std::copy(coarse_weights_v.begin(), coarse_weights_v.end(), coarse_weights_host.data());

    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

    this->primal_constraint_dofs_interface_local =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_interface_local_host);
    this->primal_constraint_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_subdomain_host);
    this->primal_constraint_offsets =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_constraint_offsets_host);
    this->coarse_dofs_local_to_global =
      Kokkos::create_mirror_view_and_copy(exec_space, coarse_dofs_local_to_global_host);
    this->corner_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, corner_dofs_subdomain_host);
    this->coarse_weights = Kokkos::create_mirror_view_and_copy(exec_space, coarse_weights_host);

    exec_space.fence();
  }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::project_to_homogeneous_constraints_interface(
    InterfaceVectorType &interface_vector) const
  {
    auto interface_vector_view = interface_vector.get_values();

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;
    const auto weights               = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              interface_vector_view(local_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const
  {
    subdomain_vector                 = 0;
    const auto interface_vector_view = interface_vector.get_values();
    auto       subdomain_vector_view = subdomain_vector.get_values();

    const auto offsets                   = this->primal_constraint_offsets;
    const auto local_constraint_dofs     = this->primal_constraint_dofs_interface_local;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface_and_scatter_to_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) =
                interface_vector_view(local_constraint_dofs(i)) - average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::project_to_homogeneous_constraints_subdomain(
    SubdomainVectorType &subdomain_vector) const
  {
    DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(),
                                               subdomain_vector.size());

    const auto offsets                   = this->primal_constraint_offsets;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += subdomain_vector_view(subdomain_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }

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

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> local_coarse_contribution(
      "local_coarse_contribution", this->n_local_coarse_dofs);

    Kokkos::parallel_for(
      "global_interface_to_coarse_local_sum",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        Number average = 0;
        for (int i = start; i < end; ++i)
          average += interface_vector_view(local_constraint_dofs(i));

        average *= weights(coarse_local_idx);

        local_coarse_contribution(coarse_local_idx) = average;
      });

    auto local_coarse_contribution_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_coarse_contribution);

    std::fill(temp_coarse_gather.begin(), temp_coarse_gather.end(), Number(0));

    for (unsigned int local_idx = 0; local_idx < this->n_local_coarse_dofs; ++local_idx)
      {
        const unsigned int global_coarse_idx  = coarse_dofs_local_to_global_vector_host[local_idx];
        temp_coarse_gather[global_coarse_idx] = local_coarse_contribution_host(local_idx);
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
      temp_local_coarse[i] = coarse_vector(coarse_dofs_local_to_global_vector_host[i]);

    // copy to host
    Kokkos::View<Number *, Kokkos::HostSpace> local_coarse_host_view(temp_local_coarse.data(),
                                                                     this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);

    auto interface_vector_view = interface_vector.get_values();

    const auto weights               = this->coarse_weights;
    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::parallel_for(
      "coarse_to_global_interface_interpolate",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const Number coarse_value =
          weights(coarse_local_idx) * local_coarse_device_view(coarse_local_idx);

        for (unsigned int i = start; i < end; ++i)
          interface_vector_view(local_constraint_dofs(i)) = coarse_value;
      });

    Kokkos::fence();

    // update globally
    interface_vector.compress(VectorOperation::insert);
  }

  template <int dim, typename Number>
  void
  BDDCPreconditioner<dim, Number>::lift_coarse_to_interface(
    InterfaceVectorType  &interface_vector,
    const Vector<Number> &coarse_vector) const
  {
    interface_vector = 0.;

    std::fill(temp_local_coarse.begin(), temp_local_coarse.end(), Number(0));

    for (unsigned int i = 0; i < this->n_local_coarse_dofs; ++i)
      temp_local_coarse[i] = coarse_vector(coarse_dofs_local_to_global_vector_host[i]);

    // copy to host
    Kokkos::View<Number *, Kokkos::HostSpace> local_coarse_host_view(temp_local_coarse.data(),
                                                                     this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);
    Kokkos::fence();

    DeviceVector<Number> interface_vector_view(interface_vector.get_values(),
                                               this->interface_vector_size);

    const auto constraint_dof_local = this->primal_constraint_dofs_interface_local;
    const auto offsets              = this->primal_constraint_offsets;
    const auto weights              = this->coarse_weights;

    Kokkos::parallel_for(
      "lift_coarse_constraints",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int local_coarse_idx) {
        const unsigned int start = offsets(local_coarse_idx);
        const unsigned int end   = offsets(local_coarse_idx + 1);

        const Number coarse_value =
          local_coarse_device_view(local_coarse_idx) * weights(local_coarse_idx);

        for (unsigned int i = start; i < end; ++i)
          {
            interface_vector_view(constraint_dof_local(i)) = coarse_value;
          }
      });
    Kokkos::fence();

    interface_vector.compress(VectorOperation::add);
  }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::compute_local_coarse_matrix(
  //   LAPACKFullMatrix<Number> &local_coarse_matrix) const
  // {
  //   const unsigned int n_local_coarse = this->n_local_coarse_dofs;
  //   local_coarse_matrix.reinit(n_local_coarse, n_local_coarse);

  //   const auto interface_dofs_subdomain = this->interface_dof_indices_subdomain;

  //   SubdomainVectorType phi, rhs, S_per_phi;
  //   phi.reinit(temp_subdomain_dst);
  //   rhs.reinit(temp_subdomain_dst);
  //   S_per_phi.reinit(temp_subdomain_dst);

  //   DeviceVector<Number> temp_subdomain_dst_view(temp_subdomain_dst.get_values(),
  //                                                temp_subdomain_dst.size());


  //   DeviceVector<Number> temp_interface_view(temp_interface.get_values(), interface_vector_size);

  //   std::vector<SubdomainVectorType> lifted_constraints(n_local_coarse);

  //   Vector<Number> e_k(n_local_coarse);


  //   for (unsigned int k = 0; k < n_local_coarse; ++k)
  //     {
  //       e_k    = 0.;
  //       e_k(k) = Number(1);

  //       this->lift_coarse_to_interface(temp_interface, e_k);

  //       temp_subdomain_dst = 0;
  //       Kokkos::parallel_for(
  //         "scatter_interface_to_subdomain",
  //         interface_dofs_subdomain.size(),
  //         KOKKOS_LAMBDA(const int) {
  //           temp_subdomain_dst_view(interface_dofs_subdomain(i)) = temp_interface_view(i);
  //         });
  //       Kokkos::fence();

  //       lifted_constraints[k] = temp_subdomain_dst;
  //     }

  //   SubdomainProjectorWrapper projector(*this);
  //   SolverControl             solver_control(1000, 1e-12 * temp_subdomain_src.l2_norm());
  //   SolverProjectedCG<SubdomainVectorType> solver(solver_control);

  //   for(unsigned int j=0; j<n_local_coarse; ++j)
  //   {
  //     phi = lifted_constraints[j];

  //     this->subdomain_op
  //   }
  // }

  // template <int dim, typename Number>
  // void
  // BDDCPreconditioner<dim, Number>::setup_coarse_matrix()
  // {
  //   const unsigned int n_global_coarse = this->n_global_coarse_dofs;
  //   coarse_matrix.reinit(n_global_coarse, n_global_coarse);

  //   LAPACKFullMatrix<Number> local_coarse_matrix;
  //   this->compute_coarse_matrix(local_coarse_matrix);


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


} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif