#ifndef portable_interface_solver_h
#define portable_interface_solver_h

#include "domain_decomposition/subdomain_dof_handler.h"
#include "operators/portable_subdomain_laplace_operator.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename number>
  class InterfaceSolver
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

  private:
    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;

    ObserverPointer<const AffineConstraints<number>> constraints;

    ObserverPointer<const SubdomainLaplaceOperator<dim, fe_degree, number>>
      subdomain_operator;
  };

  template <int dim, int fe_degree, typename number>
  InterfaceSolver<dim, fe_degree, number>::InterfaceSolver(
    const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
    const AffineConstraints<number> &constraints,
    const SubdomainLaplaceOperator<dim, fe_degree, number> &subdomain_operator)
    : subdomain_dof_handler(&subdomain_dof_handler)
    , constraints(&constraints)
    , subdomain_operator(&subdomain_operator)
  {}

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