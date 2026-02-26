#ifndef portable_interface_solver_h
#define portable_interface_solver_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include "domain_decomposition/subdomain_dof_handler.h"
#include "operators/portable_subdomain_laplace_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename number>
  class InterfaceSolver : public EnableObserverPointer
  {
  public:
    InterfaceSolver(const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
                    const AffineConstraints<number> &constraints,
                    const SubdomainLaplaceOperator<dim, fe_degree, number>
                      &subdomain_operator);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const;

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    types::global_dof_index
    m() const;
    types::global_dof_index
    n() const;

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

    void
    setup_coarse_matrix();

    void
    apply_coarse_preconditioner(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    bool
    enable_printing() const
    {
      return (this->this_subdomain == 0);
    }

  private:
    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;

    ObserverPointer<const AffineConstraints<number>> constraints;

    ObserverPointer<const SubdomainLaplaceOperator<dim, fe_degree, number>>
      subdomain_operator;

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      coarse_vector;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    LAPACKFullMatrix<number> coarse_matrix;

    mutable std::vector<number> temp_coarse_gather;
    mutable Vector<number>      temp_coarse_rhs;
    mutable Vector<number>      temp_coarse_solution;
  };

  template <int dim, int fe_degree, typename number>
  InterfaceSolver<dim, fe_degree, number>::InterfaceSolver(
    const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
    const AffineConstraints<number> &constraints,
    const SubdomainLaplaceOperator<dim, fe_degree, number> &subdomain_operator)
    : subdomain_dof_handler(&subdomain_dof_handler)
    , constraints(&constraints)
    , subdomain_operator(&subdomain_operator)
    , coarse_problem_rank(subdomain_dof_handler.n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler.n_subdomains())
    , this_subdomain(subdomain_dof_handler.get_subdomain_id())
  {
    temp_coarse_gather.resize(n_subdomains);
    temp_coarse_rhs.reinit(n_subdomains);
    temp_coarse_solution.reinit(n_subdomains);
  }

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::global_interface_to_coarse(
    Vector<number> &coarse_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));


    number subdomain_value = 0.;

    this->subdomain_operator->subdomain_interface_to_coarse(subdomain_value,
                                                            interface_vector);

    this->temp_coarse_gather = Utilities::MPI::gather(
      this->subdomain_dof_handler->get_mpi_communicator(),
      subdomain_value,
      this->coarse_problem_rank);

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        Assert(this->temp_coarse_gather.size() == this->n_subdomains,
               ExcMessage("Number of values gathered does not match number of \
                         subdomains."));
        coarse_vector = 0.;
        for (unsigned int i = 0; i < this->temp_coarse_gather.size(); ++i)
          coarse_vector[i] = this->temp_coarse_gather[i];
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::coarse_to_global_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                         &interface_vector,
    const Vector<number> &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        for (unsigned int i = 0; i < coarse_vector.size(); ++i)
          this->temp_coarse_gather[i] = coarse_vector[i];
      }

    const number subdomain_value = Utilities::MPI::scatter(
      this->subdomain_dof_handler->get_mpi_communicator(),
      this->temp_coarse_gather,
      this->coarse_problem_rank);

    this->subdomain_operator->coarse_to_subdomain_interface(interface_vector,
                                                            subdomain_value);
    interface_vector.compress(VectorOperation::add);
    interface_vector.update_ghost_values();
  }


  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::apply_coarse_preconditioner(
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

    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        this->temp_coarse_solution = 0.;

        this->coarse_matrix.vmult(this->temp_coarse_solution,
                                  this->temp_coarse_rhs);
      }

    this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  }

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::setup_coarse_matrix()
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

        this->subdomain_operator->vmult_schur(S_phi_j, phi_j);

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


  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::vmult(
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

    subdomain_operator->vmult_schur(dst, src);
  }


  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::Tvmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    this->vmult(dst, src);
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  InterfaceSolver<dim, fe_degree, number>::m() const
  {
    return this->subdomain_dof_handler->get_interface_vector_partitioner()
      ->size();
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  InterfaceSolver<dim, fe_degree, number>::n() const
  {
    return this->subdomain_dof_handler->get_interface_vector_partitioner()
      ->size();
  }


} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif