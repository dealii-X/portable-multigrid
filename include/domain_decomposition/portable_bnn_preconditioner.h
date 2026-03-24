#ifndef portable_bnn_preconditioner_h
#define portable_bnn_preconditioner_h


#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "operators/portable_subdomain_laplace_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename number>
  class BNNPreconditioner
  {
  public:
    BNNPreconditioner(
      const SchurInterfaceOperator<dim, fe_degree, number> &interface_operator,
      const SubdomainLaplaceOperator<dim, fe_degree, number>
        &subdomain_operator);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const;

    void
    project(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    void
    balance(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    void
    setup_coarse_matrix();

    void
    coarse_to_global_interface(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                           &interface_vector,
      const Vector<number> &coarse_vector) const;

    void
    global_interface_to_coarse(
      Vector<number> &coarse_vector,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &interface_vector) const;

  private:
    ObserverPointer<const SchurInterfaceOperator<dim, fe_degree, number>>
      interface_operator;
    ObserverPointer<const SubdomainLaplaceOperator<dim, fe_degree, number>>
                                                    subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;

    LAPACKFullMatrix<number> coarse_matrix;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &interface_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      temp_interface;

    mutable std::vector<number> temp_coarse_gather;
    mutable Vector<number>      temp_coarse_rhs;
    mutable Vector<number>      temp_coarse_solution;
  };

  template <int dim, int fe_degree, typename number>
  BNNPreconditioner<dim, fe_degree, number>::BNNPreconditioner(
    const SchurInterfaceOperator<dim, fe_degree, number>   &interface_operator,
    const SubdomainLaplaceOperator<dim, fe_degree, number> &subdomain_operator)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , coarse_problem_rank(subdomain_dof_handler->n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(
        subdomain_operator.get_interface_dof_indices_subdomain())
  {
    temp_interface.reinit(
      this->subdomain_dof_handler->get_interface_vector_partitioner());

    temp_coarse_gather.resize(n_subdomains);
    temp_coarse_rhs.reinit(n_subdomains);
    temp_coarse_solution.reinit(n_subdomains);
  }


  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(
      dst.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    this->interface_operator->neumann_solve_subdomain(dst, src);
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
  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::project(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(
      dst.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(
      src.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    temp_interface = 0.;

    // dst = S*tmp
    this->interface_operator->vmult(temp_interface, src);

    // tmp = R0^T*S_0^{-1}*R0*dst
    this->balance(dst, temp_interface);

    dst.sadd(-1., src);
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
  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::balance(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(
      dst.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(
      src.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    this->temp_coarse_rhs = 0.;

    // project from global interface to coarse space
    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    // solve coarse problem
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        this->temp_coarse_solution = 0.;

        this->coarse_matrix.vmult(this->temp_coarse_solution,
                                  this->temp_coarse_rhs);
      }

    // project back to the global interface space
    this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  }


  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::global_interface_to_coarse(
    Vector<number> &coarse_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));


    interface_vector.update_ghost_values();

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_vector.size());

    DeviceVector<number> weights_view(interface_weights.get_values(),
                                      interface_weights.size());

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
    this->temp_coarse_gather = Utilities::MPI::gather(
      this->subdomain_dof_handler->get_mpi_communicator(),
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

  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::coarse_to_global_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                         &interface_vector,
    const Vector<number> &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_vector.size());

    DeviceVector<number> weights_view(interface_weights.get_values(),
                                      interface_weights.size());


    // copy coarse Vector to std::vector for MPi::scatter
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        for (unsigned int i = 0; i < coarse_vector.size(); ++i)
          this->temp_coarse_gather[i] = coarse_vector[i];
      }

    // retrieve subdomain coarse value (i.e., mean value)
    const number subdomain_coarse_value = Utilities::MPI::scatter(
      this->subdomain_dof_handler->get_mpi_communicator(),
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

  template <int dim, int fe_degree, typename number>
  void
  BNNPreconditioner<dim, fe_degree, number>::setup_coarse_matrix()
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

        this->interface_operator->vmult(S_phi_j, phi_j);

        this->global_interface_to_coarse(coarse_column, S_phi_j);

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