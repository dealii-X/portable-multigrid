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

    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      temp_interface1, temp_interface2;

    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      temp_coarse1, temp_coarse2;
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
  {
    temp_interface1.reinit(
      this->subdomain_dof_handler->get_interface_vector_partitioner());
    temp_interface2.reinit(temp_interface1);

    temp_coarse1.reinit(this->subdomain_dof_handler->n_subdomains());
    temp_coarse2.reinit(temp_coarse1);
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

    // temp_interface1.reinit(dst);
    // temp_interface2.reinit(dst);

    // // local Neumann-Neumann part
    // this->subdomain_operator->neumann_solve_subdomain(temp_interface1, src);

    // // tmp2 = S*tmp1
    // this->subdomain_operator->vmult_schur(dst, temp_interface1);

    // // tmp2 = src-S*tmp1
    // temp_interface2.sadd(-1., 1., src);

    // // dst = R0^T*S_0^{-1}*R0*tmp2
    // this->interface_solver->apply_coarse_preconditioner(dst,
    // temp_interface2);

    // // dst = src - dst
    // dst.add(1., temp_interface2);

    this->subdomain_operator->neumann_solve_subdomain(dst, src);
  }


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

    temp_interface1 = 0.;
    temp_interface2 = 0.;


    // src.print(std::cout);

    // dst = S*tmp
    this->subdomain_operator->vmult_schur(temp_interface1, src);


    // temp_interface1.print(std::cout);


    // tmp = R0^T*S_0^{-1}*R0*dst
    this->interface_solver->apply_coarse_preconditioner(dst, temp_interface1);

    // temp_interface1.print(std::cout);



    // dst.update_ghost_values();
    // dst = temp_interface1;

    // dst -= temp_interface2;

    // temp_interface2.print(std::cout);

    dst.sadd(-1., src);

    // dst.print(std::cout);
  }



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

    this->interface_solver->apply_coarse_preconditioner(dst, src);
  }


  //   template <int dim, int fe_degree, typename number>
  //   void
  //   BNNPreconditioner<dim, fe_degree, number>::vmult(
  //     LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
  //     const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
  //     &src) const
  //   {
  //     Assert(
  //       dst.get_partitioner() ==
  //         this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //       ExcMessage(
  //         "This function expects a vector initialized by
  //         SubdomainDoFHandler's
  //              interface vector partitioner."));
  //     Assert(
  //       src.get_partitioner() ==
  //         this->subdomain_dof_handler->get_interface_vector_partitioner(),
  //       ExcMessage(
  //         "This function expects a vector initialized by
  //         SubdomainDoFHandler's
  //             interface vector partitioner."));

  //     LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
  //     u_coarse1(
  //       src.get_partitioner()),
  //       u_coarse2(src.get_partitioner()), r_bal(src.get_partitioner()),
  //       u_loc(src.get_partitioner()), tmp(src.get_partitioner());


  //     // u_coarse1 = R0^T*S0^{-1}*R0*src =Q*src
  //     this->interface_solver->apply_coarse_preconditioner(u_coarse1, src);


  //     // tmp = S*u_coarse1 = S*Q*src
  //     this->subdomain_operator->vmult_schur(tmp, u_coarse1);

  //     // r_bal = (I-Q*S)*src
  //     r_bal.add(1., src, -1., tmp);

  //     // r_bal.print(std::cout);

  //     // u_loc = D*Si^{-1}*D*(I-Q*S)*src = P_local*(I-Q*S)*src
  //     this->subdomain_operator->neumann_solve_subdomain(u_loc, r_bal);

  //     // u_loc.update_ghost_values();

  //     // tmp = S*P_local*(I-Q*S)*src
  //     this->subdomain_operator->vmult_schur(tmp, u_loc);

  //     // tmp = Q*S*P_local*(I-Q*S)*src
  //     this->interface_solver->apply_coarse_preconditioner(u_coarse2, tmp);

  //     // dst = (I -Q*S)*P_local*(I-Q*S)*src
  //     dst.add(1., u_loc, -1., u_coarse2);

  //     // dst.print(std::cout);

  //     // dst = u_coarse1 + (I -Q*S)*P_local*(I-Q*S)*src
  //     dst.add(1., u_coarse1);

  //     // dst.print(std::cout);

  //     src.zero_out_ghost_values();
  //   }



} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif