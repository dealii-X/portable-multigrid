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
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &coarse_vector) const;

    void
    global_interface_to_coarse(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &coarse_vector,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &interface_vector) const;

    void
    setup_coarse_matrix();

    void
    apply_coarse_preconditioner(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

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
  {}

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::global_interface_to_coarse(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &coarse_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));


    number subdomain_value = 0.;

    this->subdomain_operator->subdomain_interface_to_coarse(subdomain_value,
                                                            interface_vector);

    std::vector<number> all_coarse_values = Utilities::MPI::gather(
      this->subdomain_dof_handler->get_mpi_communicator(),
      subdomain_value,
      this->coarse_problem_rank);

    LinearAlgebra::ReadWriteVector<number> rw_vector(
      coarse_vector.locally_owned_elements());
    rw_vector.import_elements(coarse_vector, VectorOperation::insert);
    LinearAlgebra::distributed::Vector<number, MemorySpace::Host> coarse_host(
      coarse_vector.get_partitioner());
    coarse_host.import_elements(rw_vector, VectorOperation::insert);


    if (this->this_subdomain == this->coarse_problem_rank)
      {
        Assert(all_coarse_values.size() == this->n_subdomains,
               ExcMessage("Number of values gathered does not match number of \
                         subdomains."));

        for (unsigned int i = 0; i < all_coarse_values.size(); ++i)
          coarse_host[i] = all_coarse_values[i];
      }

    rw_vector.import_elements(coarse_host, VectorOperation::insert);
    coarse_vector.import_elements(rw_vector, VectorOperation::insert);
  }

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::coarse_to_global_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &interface_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    LinearAlgebra::ReadWriteVector<number> rw_vector(
      coarse_vector.locally_owned_elements());
    rw_vector.import_elements(coarse_vector, VectorOperation::insert);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Host> coarse_host(
      coarse_vector.get_partitioner());
    coarse_host.import_elements(rw_vector, VectorOperation::insert);

    std::vector<number> all_coarse_values;
    if (this->this_subdomain == this->coarse_problem_rank)
      {
        all_coarse_values.resize(this->n_subdomains);
        for (unsigned int i = 0; i < all_coarse_values.size(); ++i)
          all_coarse_values[i] = coarse_host[i];
      }

    const number subdomain_value = Utilities::MPI::scatter(
      this->subdomain_dof_handler->get_mpi_communicator(),
      all_coarse_values,
      this->coarse_problem_rank);

    // for (unsigned int i = 0; i < all_coarse_values.size(); i++)
    //   {
    //     std::cout << "On subdomain " << this_subdomain << ": "
    //               << subdomain_value;
    //     std::cout << std::endl;
    //   }

    interface_vector = 0.;

    this->subdomain_operator->coarse_to_subdomain_interface(interface_vector,
                                                            subdomain_value);
    interface_vector.compress(VectorOperation::add);
    interface_vector.update_ghost_values();

    // interface_vector.print(std::cout);

    //     if (this->subdomain_dof_handler->get_subdomain_id() == 3)
    //   for (unsigned int i = 0; i < all_coarse_values.size(); i++)
    //     std::cout << all_coarse_values[i] << " ";
    // std::cout << std::endl;
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

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> coarse_rhs(
      this->n_subdomains);

    this->global_interface_to_coarse(coarse_rhs, src);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      coarse_solution(this->n_subdomains);

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        LinearAlgebra::distributed::Vector<number, MemorySpace::Host>
          coarse_rhs_host(this->n_subdomains),
          coarse_solution_host(this->n_subdomains);

        LinearAlgebra::ReadWriteVector<number> rw_vector(
          coarse_rhs_host.size());
        rw_vector.import_elements(coarse_rhs, VectorOperation::insert);
        coarse_rhs_host.import_elements(rw_vector, VectorOperation::insert);

        Vector<number> coarse_solution_local(this->n_subdomains),
          coarse_rhs_local(this->n_subdomains);
        for (unsigned int i = 0; i < this->n_subdomains; ++i)
          coarse_rhs_local(i) = coarse_rhs_host[i];

        this->coarse_matrix.vmult(coarse_solution_local, coarse_rhs_local);

        for (unsigned int i = 0; i < this->n_subdomains; ++i)
          coarse_solution_host[i] = coarse_solution_local(i);

        rw_vector.import_elements(coarse_solution_host,
                                  VectorOperation::insert);
        coarse_solution.import_elements(rw_vector, VectorOperation::insert);
      }

    this->coarse_to_global_interface(dst, coarse_solution);
  }

  template <int dim, int fe_degree, typename number>
  void
  InterfaceSolver<dim, fe_degree, number>::setup_coarse_matrix()
  {
    if (this->this_subdomain == this->coarse_problem_rank)
      coarse_matrix.reinit(this->n_subdomains, this->n_subdomains);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> phi_j(
      this->subdomain_dof_handler->get_interface_vector_partitioner()),
      S_phi_j(this->subdomain_dof_handler->get_interface_vector_partitioner()),
      e_j(this->n_subdomains), coarse_column(this->n_subdomains);


    LinearAlgebra::distributed::Vector<number, MemorySpace::Host> e_j_host(
      this->n_subdomains),
      coarse_column_host(this->n_subdomains);

    for (unsigned int j = 0; j < this->n_subdomains; ++j)
      {
        e_j_host = 0.;

        if (this->this_subdomain == this->coarse_problem_rank)
          e_j_host[j] = 1.;
        LinearAlgebra::ReadWriteVector<number> rw_vector(e_j_host.size());
        rw_vector.import_elements(e_j_host, VectorOperation::insert);
        e_j.import_elements(rw_vector, VectorOperation::insert);

        this->coarse_to_global_interface(phi_j, e_j);

        this->subdomain_operator->vmult_schur(S_phi_j, phi_j);

        this->global_interface_to_coarse(coarse_column, S_phi_j);

        // coarse_column.print(std::cout);

        rw_vector.reinit(coarse_column.size());
        rw_vector.import_elements(coarse_column, VectorOperation::insert);
        coarse_column_host.import_elements(rw_vector, VectorOperation::insert);


        if (this->this_subdomain == this->coarse_problem_rank)
          for (unsigned int i = 0; i < this->n_subdomains; ++i)
            coarse_matrix(i, j) = coarse_column_host[i];
      }
    // MPI_Barrier(this->subdomain_dof_handler->get_mpi_communicator());

    // if (this->this_subdomain == this->coarse_problem_rank)
    //   coarse_matrix.print_formatted(std::cout, 3, true, 0, "0.00e+00");

    if (this->this_subdomain == this->coarse_problem_rank)
      {
        coarse_matrix.compute_inverse_svd(1e-12);

        std::cout << "Singular values of the coarse matrix: " << std::endl;
        for (unsigned int i = 0; i < this->n_subdomains; ++i)
          std::cout << coarse_matrix.singular_value(i) << " ";
        std::cout << std::endl;
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