#ifndef portable_subdomain_laplace_operator_base_h
#define portable_subdomain_laplace_operator_base_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "domain_decomposition/subdomain_dof_handler.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename Number>
  class SubdomainLaplaceOperatorBase : public EnableObserverPointer
  {
  public:
    virtual void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    virtual void
    project(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const = 0;

    virtual void
    vmult_plain(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    /**
     * Block vmult_plain(): applies vmult_plain() independently to each of
     * n_rhs blocks of dst/src, which must be sized n_rhs * (this
     * operator's dof count) and laid out as n_rhs blocks of that size
     * each -- the same convention bk3_kokkos_kernel_block.h's
     * KokkosKernelBlock()/SolverBlockCG use.
     */
    virtual void
    vmult_plain_block(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      const unsigned int                                                     n_rhs) const = 0;

    /**
     * Block project(): applies project() independently to each of n_rhs
     * blocks of vec, same layout convention as vmult_plain_block().
     */
    virtual void
    project_block(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec,
                  const unsigned int n_rhs) const = 0;

    virtual void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    virtual void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const = 0;

    virtual void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    virtual void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    // virtual void
    // vmult_bddc_preconditioner(
    //   LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    //   const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    virtual void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;

    virtual void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const = 0;

    virtual void
    compute_diagonal() = 0;

    virtual std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const = 0;

    virtual std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const = 0;

    virtual types::global_dof_index
    m() const = 0;

    virtual types::global_dof_index
    n() const = 0;

    virtual Number
    el(const types::global_dof_index row, const types::global_dof_index col) const = 0;

    virtual const MatrixFree<dim, Number> &
    get_matrix_free() const = 0;

    virtual const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const = 0;

    virtual const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const = 0;


    virtual const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const = 0;

    virtual const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const = 0;
  };

  class SubdomainOperatorDispatchFactory
  {
  public:
    static constexpr unsigned int max_degree = 4;

    template <typename OperatorRunner>
    static bool
    dispatch(const unsigned int runtime_degree, OperatorRunner &runner)
    {
      return recursive_dispatch<OperatorRunner, max_degree>(runtime_degree, runner);
    }

  private:
    template <typename OperatorRunner, unsigned int degree>
    static bool
    recursive_dispatch(const unsigned int runtime_degree, OperatorRunner &runner)
    {
      if (runtime_degree == degree)
        {
          runner.template run<degree>();
          return true;
        }
      else if constexpr (degree > 1)
        {
          return recursive_dispatch<OperatorRunner, degree - 1>(runtime_degree, runner);
        }
      else
        {
          return false;
        }
    }
  };


} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif