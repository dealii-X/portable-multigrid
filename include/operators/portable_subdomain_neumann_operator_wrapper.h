#ifndef portable_subdomain_neumann_operator_wrapper_h
#define portable_subdomain_neumann_operator_wrapper_h

#include <deal.II/base/observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"
#include "operators/portable_subdomain_laplace_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename number>
  class SubdomainNeumannOperatorWrapper : public SubdomainLaplaceOperatorBase<dim, number>
  {
  public:
    SubdomainNeumannOperatorWrapper(
      const SubdomainLaplaceOperatorBase<dim, number> &dirichlet_operator)
      : dirichlet_operator(&dirichlet_operator)
    {}

    void
    vmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_neumann(dst, src);
    }

    void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override
    {
      dirichlet_operator->vmult_dummy(dst, src, ghost_exchange_on, computation_on);
    }

    void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_interface_cell_range(dst, src);
    }

    void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_neumann(dst, src);
    }

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override
    {
      this->vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const override
    {
      dirichlet_operator->initialize_dof_vector(vec);
    }

    void
    compute_diagonal() override
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override
    {
      return dirichlet_operator->get_matrix_diagonal_inverse_neumann();
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const override
    {
      return dirichlet_operator->get_matrix_diagonal_inverse_neumann();
    }

    types::global_dof_index
    m() const override
    {
      return dirichlet_operator->m();
    }

    types::global_dof_index
    n() const override
    {
      return dirichlet_operator->n();
    }

    number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      (void)col;
      Assert(row == col, ExcNotImplemented());

      const auto &inverse_diagonal_neumann =
        dirichlet_operator->get_matrix_diagonal_inverse_neumann();

      Assert(inverse_diagonal_neumann.get() != nullptr && inverse_diagonal_neumann->m() > 0,
             ExcNotInitialized());
      return 1.0 / (*inverse_diagonal_neumann)(row, row);
    }

    const MatrixFree<dim, number> &
    get_matrix_free() const override
    {
      return dirichlet_operator->get_matrix_free();
    }

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override
    {
      return dirichlet_operator->get_vector_partitioner();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_interface_dof_indices_subdomain();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_physical_boundary_dof_indices_subdomain();
    }

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override
    {
      return dirichlet_operator->get_subdomain_dof_handler();
    }


  private:
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, number>> dirichlet_operator;
  };
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif