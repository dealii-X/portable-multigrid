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

  template <int dim, typename number>
  class BNNPreconditioner
  {
  public:
    BNNPreconditioner(const SchurInterfaceOperator<dim, number>       &interface_operator,
                      const SubdomainLaplaceOperatorBase<dim, number> &subdomain_operator);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    project(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
            const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    balance(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
            const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    balance_dummy(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
                  const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
                  const bool computation_on,
                  const bool communication_on) const;

    void
    balance_and_vmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &S_per_dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    setup_coarse_matrix();

    void
    coarse_to_global_interface(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
      const Vector<number>                                             &coarse_vector) const;



    void
    coarse_to_global_interface_and_S_update(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &S_per_interface_vector,
      const Vector<number>                                             &coarse_vector) const;


    void
    vmult_enhanced(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &z,
                   LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &s_tilde,
                   const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &r) const;

    void
    global_interface_to_coarse(
      Vector<number>                                                         &coarse_vector,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector)
      const;

    void
    reset_timings() const;


    const std::array<double, 4> &
    get_timings() const;

    void
    vmult_interface(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

  private:
    ObserverPointer<const SchurInterfaceOperator<dim, number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    LAPACKFullMatrix<number> coarse_matrix;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default> temp_interface, z0,
      S_z0;

    std::vector<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>
      S_per_coarse_basis_functions;

    mutable std::vector<number> temp_coarse_gather;
    mutable std::vector<number> temp_coarse_broadcast;

    mutable Vector<number> temp_coarse_rhs;
    mutable Vector<number> temp_coarse_solution;

    /**
     * timings[0] = Dirichler solve
     * timings[1] = Neumann solve
     * timings[2] = coarse solve
     * timings[3] = projection step
     */
    mutable std::array<double, 4> timings;
  };

  template <int dim, typename number>
  BNNPreconditioner<dim, number>::BNNPreconditioner(
    const SchurInterfaceOperator<dim, number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, number> &subdomain_operator)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , coarse_problem_rank(subdomain_dof_handler->n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
  {
    temp_interface.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    S_per_coarse_basis_functions.resize(n_subdomains);

    for (unsigned int i = 0; i < n_subdomains; ++i)
      S_per_coarse_basis_functions[i].reinit(temp_interface);

    z0.reinit(temp_interface);
    S_z0.reinit(temp_interface);

    temp_coarse_gather.resize(n_subdomains);
    temp_coarse_rhs.reinit(n_subdomains);
    temp_coarse_solution.reinit(n_subdomains);
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::reset_timings() const
  {
    for (unsigned int i = 0; i < timings.size(); ++i)
      timings[i] = 0.;
  }

  template <int dim, typename number>
  const std::array<double, 4> &
  BNNPreconditioner<dim, number>::get_timings() const
  {
    return timings;
  }


  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_enhanced(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &z,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &s_tilde,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &r) const
  {
    Assert(z.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(r.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));


    // local NN preconditioner
    this->vmult(z, r);

    this->vmult_interface(s_tilde, z);

    temp_interface = r;
    temp_interface -= s_tilde;

    Kokkos::fence();
    Timer time;

    this->balance_and_vmult(z0, S_z0, temp_interface);

    Kokkos::fence();
    timings[2] += time.wall_time();

    s_tilde += S_z0;
    z += z0;
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(
      dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    Kokkos::fence();
    Timer time;
    this->interface_operator->neumann_solve_subdomain(dst, src);

    Kokkos::fence();
    timings[1] += time.wall_time();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(
      dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    Kokkos::fence();
    Timer time;
    this->interface_operator->vmult(dst, src);

    Kokkos::fence();
    timings[0] += time.wall_time();
  }

  /**
   * Schwarz projection step Id-(R_0^T*S_0^{-1}*R_0)*S :
   *    1. Apply interface operator S (maps values into flux)
   *    2. balance (i.e, apply R_0^T*S_0^{-1}*R_0) to retrive mean value
   *    3. Id-(R_0^T*S_0^{-1}*R_0)*S - gives compatible vector for subdomain
   * Neumann solve Result: retrieved values in the kernel space (i.e., mean
   * values) (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e.
   * compatible for the subdomain Neumann solve
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::project(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    temp_interface = 0.;

    Kokkos::fence();
    Timer time;
    // dst = S*tmp
    this->interface_operator->vmult(temp_interface, src);

    Kokkos::fence();
    const double time_dirichlet = time.wall_time();
    timings[0] += time_dirichlet;

    time.restart();
    // tmp = R0^T*S_0^{-1}*R0*dst
    this->balance(dst, temp_interface);

    Kokkos::fence();
    timings[2] += time.wall_time();

    dst.sadd(-1., src);

    Kokkos::fence();
    timings[3] += time.wall_time() + time_dirichlet;
  }

  /**
   * Balancing step R_0^T*S_0^{-1}*R_0:
   *    1. project from global interface to coarse space
   *    2. solve coarse problem
   *    3. project back to the global interface
   * Result: retrieved values in the kernel space (i.e., mean values)
   * (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e. compatible for
   * the subdomain Neumann solve
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::balance(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    this->temp_coarse_rhs = 0.;

    // project from global interface to coarse space
    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    // solve coarse problem
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        this->temp_coarse_solution = 0.;

        this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
      }

    // project back to the global interface space
    this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::balance_and_vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &S_per_dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    this->temp_coarse_rhs = 0.;

    // project from global interface to coarse space
    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    // solve coarse problem
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        this->temp_coarse_solution = 0.;

        this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
      }

    // project back to the global interface space
    this->coarse_to_global_interface_and_S_update(dst, S_per_dst, this->temp_coarse_solution);
  }


  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::balance_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool                                                              computation_on,
    const bool                                                              communication_on) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    this->temp_coarse_rhs = 0.;

    // project from global interface to coarse space
    if (communication_on)
      this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    if (computation_on)
      {
        // solve coarse problem
        if (this->this_subdomain == this->coarse_problem_rank)
          {
            this->temp_coarse_solution = 0.;

            this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
          }
      }

    // project back to the global interface space
    if (communication_on)
      this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  }


  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::global_interface_to_coarse(
    Vector<number>                                                         &coarse_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));


    interface_vector.update_ghost_values();

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_vector.size());

    DeviceVector<number> weights_view(interface_weights.get_values(), interface_weights.size());

    // retrieve subdomain coarse value by the interface weighted sum
    number subdomain_coarse_value = 0.;
    Kokkos::parallel_reduce(
      "global_interface_to_coarse",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i, number &coarse_value) {
        coarse_value += interface_vector_view(i) * weights_view(i);
      },
      subdomain_coarse_value);

    // gather all coarse values
    this->temp_coarse_gather =
      Utilities::MPI::gather(this->subdomain_dof_handler->get_mpi_communicator(),
                             subdomain_coarse_value,
                             this->coarse_problem_rank);

    // copy coarse std::vector to Vector
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        Assert(this->temp_coarse_gather.size() == this->n_subdomains,
               ExcMessage("Number of values gathered does not match number of \
                         subdomains."));
        coarse_vector = 0.;
        for (unsigned int i = 0; i < this->temp_coarse_gather.size(); ++i)
          coarse_vector[i] = this->temp_coarse_gather[i];
      }

    interface_vector.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::coarse_to_global_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
    const Vector<number>                                             &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_vector.size());

    DeviceVector<number> weights_view(interface_weights.get_values(), interface_weights.size());


    // copy coarse Vector to std::vector for MPi::scatter
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        for (unsigned int i = 0; i < coarse_vector.size(); ++i)
          this->temp_coarse_gather[i] = coarse_vector[i];
      }

    // retrieve subdomain coarse value (i.e., mean value)
    const number subdomain_coarse_value =
      Utilities::MPI::scatter(this->subdomain_dof_handler->get_mpi_communicator(),
                              this->temp_coarse_gather,
                              this->coarse_problem_rank);

    // propagate coarse value to the interface by applying weights
    interface_vector = 0.;
    Kokkos::parallel_for(
      "SubdomainLaplaceOperator::coarse_to_subdomain_interface",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i) {
        interface_vector_view(i) = subdomain_coarse_value * weights_view(i);
      });

    // condense
    interface_vector.compress(VectorOperation::add);
    interface_vector.update_ghost_values();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::coarse_to_global_interface_and_S_update(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &S_per_interface_vector,
    const Vector<number>                                             &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_vector.size());


    DeviceVector<number> weights_view(interface_weights.get_values(), interface_weights.size());

    // copy coarse Vector to std::vector for MPi::scatter
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        for (unsigned int i = 0; i < coarse_vector.size(); ++i)
          this->temp_coarse_gather[i] = coarse_vector[i];
      }

    // retrieve subdomain coarse value (i.e., mean value)
    // const number subdomain_coarse_value = Utilities::MPI::scatter(
    //   this->subdomain_dof_handler->get_mpi_communicator(),
    //   this->temp_coarse_gather,
    //   this->coarse_problem_rank);

    temp_coarse_broadcast =
      Utilities::MPI::broadcast(this->subdomain_dof_handler->get_mpi_communicator(),
                                this->temp_coarse_gather,
                                this->coarse_problem_rank);

    const number subdomain_coarse_value = temp_coarse_broadcast[this_subdomain];

    // propagate coarse value to the interface by applying weights
    interface_vector = 0.;
    Kokkos::parallel_for(
      "SubdomainLaplaceOperator::coarse_to_subdomain_interface",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i) {
        interface_vector_view(i) = subdomain_coarse_value * weights_view(i);
      });

    // condense
    interface_vector.compress(VectorOperation::add);

    S_per_interface_vector = 0;
    for (unsigned int i = 0; i < n_subdomains; ++i)
      S_per_interface_vector.add(temp_coarse_broadcast[i], S_per_coarse_basis_functions[i]);

    S_per_interface_vector.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::setup_coarse_matrix()
  {
    if (this->this_subdomain == this->coarse_problem_rank)
      coarse_matrix.reinit(this->n_subdomains, this->n_subdomains);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> phi_j(
      this->subdomain_dof_handler->get_interface_vector_partitioner()),
      S_phi_j(this->subdomain_dof_handler->get_interface_vector_partitioner());

    Vector<number> e_j(this->n_subdomains), coarse_column(this->n_subdomains);

    for (unsigned int j = 0; j < this->n_subdomains; ++j)
      {
        e_j = 0.;

        if (this->this_subdomain == this->coarse_problem_rank)
          e_j[j] = 1.;

        this->coarse_to_global_interface(phi_j, e_j);

        this->interface_operator->vmult(S_per_coarse_basis_functions[j], phi_j);

        this->global_interface_to_coarse(coarse_column, S_per_coarse_basis_functions[j]);

        if (this->this_subdomain == this->coarse_problem_rank)
          for (unsigned int i = 0; i < this->n_subdomains; ++i)
            coarse_matrix(i, j) = coarse_column[i];
      }

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        coarse_matrix.compute_inverse_svd(1e-12);

        // std::cout << "Singular values of the coarse matrix: " << std::endl;
        // for (unsigned int i = 0; i < this->n_subdomains; ++i)
        //   std::cout << coarse_matrix.singular_value(i) << " ";
        // std::cout << std::endl;
      }
  }

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif