#ifndef portable_subdomain_bddc_operator_wrapper_h
#define portable_subdomain_bddc_operator_wrapper_h

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

  template <int dim, typename Number>
  class SubdomainBDDCOperatorWrapper : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    SubdomainBDDCOperatorWrapper(
      const SubdomainLaplaceOperatorBase<dim, Number> &dirichlet_operator)
      : dirichlet_operator(&dirichlet_operator)
      , inverse_diagonal_entries(
          std::make_shared<
            DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>())
    {}

    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);
    }


    void
    vmult_plain(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);

    }

    void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override
    {
      (void)dst;
      (void)src;
      (void)ghost_exchange_on;
      (void)computation_on;

      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_interface_cell_range(dst, src);
    }

    void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_neumann(dst, src);
    }

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      this->vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override
    {
      dirichlet_operator->initialize_dof_vector(vec);
    }


    void
    compute_diagonal() override
    {
      inverse_diagonal_entries->reinit(
        dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

      // Number *raw_diagonal = inverse_diagonal_entries->get_vector().get_values();

      // Kokkos::parallel_for(
      //   "SubdomainBDDCOperatorWrapper::set_constrained_digonal_dofs_to_ones",
      //   subdomain_corner_dofs.size(),
      //   KOKKOS_LAMBDA(int i) { raw_diagonal[subdomain_corner_dofs(i)] = Number(1.); });
    }


    // void
    // compute_diagonal(Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
    //                    subdomain_corner_dofs) override
    // {
    //   inverse_diagonal_entries->reinit(
    //     dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

    //   Number *raw_diagonal = inverse_diagonal_entries->get_vector().get_values();

    //   Kokkos::parallel_for(
    //     "SubdomainBDDCOperatorWrapper::set_constrained_digonal_dofs_to_ones",
    //     subdomain_corner_dofs.size(),
    //     KOKKOS_LAMBDA(int i) { raw_diagonal[subdomain_corner_dofs(i)] = Number(1.); });
    // }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override
    {
      return this->inverse_diagonal_entries;
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
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

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      (void)col;
      Assert(row == col, ExcNotImplemented());

      Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
             ExcNotInitialized());

      return 1.0 / (*inverse_diagonal_entries)(row, row);
    }

    const MatrixFree<dim, Number> &
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
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> dirichlet_operator;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries;
  };
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif