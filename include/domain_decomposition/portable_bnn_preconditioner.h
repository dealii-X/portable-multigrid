#ifndef portable_bnn_preconditioner_h
#define portable_bnn_preconditioner_h


#include "domain_decomposition/portable_interface_solver.h"
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
      const SubdomainDoFHandler<dim>                &subdomain_dof_handler,
      const InterfaceSolver<dim, fe_degree, number> &interface_solver,
      const SubdomainLaplaceOperator<dim, fe_degree, number>
        &subdomain_operator);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const;

  private:
    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;
    ObserverPointer<const InterfaceSolver<dim, fe_degree, number>>
      interface_solver;
    ObserverPointer<const SubdomainLaplaceOperator<dim, fe_degree, number>>
      subdomain_operator;

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                       coarse_vector;
    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;
  };

  template <int dim, int fe_degree, typename number>
  BNNPreconditioner<dim, fe_degree, number>::BNNPreconditioner(
    const SubdomainDoFHandler<dim>                &subdomain_dof_handler,
    const InterfaceSolver<dim, fe_degree, number> &interface_solver,
    const SubdomainLaplaceOperator<dim, fe_degree, number> &subdomain_operator)
    : subdomain_dof_handler(&subdomain_dof_handler)
    , interface_solver(&interface_solver)
    , subdomain_operator(&subdomain_operator)
    , coarse_problem_rank(subdomain_dof_handler.n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler.n_subdomains())
    , this_subdomain(subdomain_dof_handler.get_subdomain_id())
  {}



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
        "This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(
      src.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> u_coarse1(
      src.get_partitioner()),
      u_coarse2(src.get_partitioner()), r_bal(src.get_partitioner()),
      u_loc(src.get_partitioner()), tmp(src.get_partitioner());


    // u_coarse1 = R0^T*S0^{-1}*R0*src =Q*src
    this->interface_solver->apply_coarse_preconditioner(u_coarse1, src);


    // tmp = S*u_coarse1 = S*Q*src
    this->subdomain_operator->vmult_schur(tmp, u_coarse1);


    // r_bal = (I-Q*S)*src
    r_bal.add(1., src, -1., tmp);
    // r_bal.update_ghost_values();


    // rbal = D*(I-Q*S)*src
    this->subdomain_operator->apply_interface_weights(r_bal);

    // r_bal.update_ghost_values();

    // u_loc = Si^{-1}*D*(I-Q*S)*src
    this->subdomain_operator->neumann_solve_subdomain(u_loc, r_bal);


    // u_loc = D*Si^{-1}*D*(I-Q*S)*src = P_local*(I-Q*S)*src
    this->subdomain_operator->apply_interface_weights(u_loc);

    u_loc.compress(VectorOperation::add);
    // u_loc.update_ghost_values();

    // u_loc.print(std::cout);


    // tmp = S*P_local*(I-Q*S)*src
    this->subdomain_operator->vmult_schur(tmp, u_loc);


    // tmp = Q*S*P_local*(I-Q*S)*src
    this->interface_solver->apply_coarse_preconditioner(u_coarse2, tmp);



    // dst = (I -Q*S)*P_local*(I-Q*S)*src
    dst.add(1., u_loc, -1., u_coarse2);

    // dst.update_ghost_values();
    // dst.print(std::cout);
    // dst = u_coarse1 + (I -Q*S)*P_local*(I-Q*S)*src
    dst.add(1., u_coarse1);



    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    // dst.print(std::cout);
  }



} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif