#ifndef portable_block_vcycle_adapters_h
#define portable_block_vcycle_adapters_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/observer_pointer.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <Kokkos_Core.hpp>

#include <memory>

#include "base/portable_mg_transfer_base.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  /**
   * Thin adapters that let the *existing*, unmodified
   * SolverBlockCG/ProjectedChebyshevSmoother/SubdomainVCycleMultigrid
   * templates (all of which only ever call plain-looking vmult()/
   * project()/restrict_and_add()/prolongate_and_add()/
   * initialize_dof_vector() -- no n_rhs parameter anywhere in their own
   * signatures) drive the block-shaped BDDC coarse solve. Each adapter
   * captures n_rhs at construction and forwards to the corresponding
   * `_block(..., n_rhs)` method added to the concrete operator/transfer
   * classes (see vmult_plain_block()/project_block() on
   * SubdomainBDDCOperator/SubdomainLaplaceOperator and
   * restrict_and_add_block()/prolongate_and_add_block() on
   * PolynomialTransfer), rather than threading n_rhs through every
   * virtual signature in the codebase.
   */
  template <int dim, typename Number>
  class BlockBDDCOperatorAdapter : public EnableObserverPointer
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    BlockBDDCOperatorAdapter(const SubdomainBDDCOperator<dim, Number> &op,
                             const unsigned int                               n_rhs)
      : op(&op)
      , n_rhs(n_rhs)
    {}

    // Ahat_block = Pi * A * Pi applied blockwise: matches
    // SubdomainBDDCOperator::vmult()'s own
    // vmult_plain()+project() composition, just on n_rhs blocks at once.
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      op->vmult_plain_block(dst, src, n_rhs);
      op->project_block(dst, n_rhs);
    }

    void
    project(VectorType &dst) const
    {
      op->project_block(dst, n_rhs);
    }

    void
    initialize_dof_vector(VectorType &vec) const
    {
      VectorType single;
      op->initialize_dof_vector(single);
      vec.reinit(static_cast<typename VectorType::size_type>(n_rhs) * single.size());
    }

  private:
    ObserverPointer<const SubdomainBDDCOperator<dim, Number>> op;
    const unsigned int                                               n_rhs;
  };

  template <int dim, typename Number>
  class BlockTransferAdapter : public EnableObserverPointer
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    BlockTransferAdapter(const MGTransferBase<dim, Number> &transfer, const unsigned int n_rhs)
      : transfer(&transfer)
      , n_rhs(n_rhs)
    {}

    void
    restrict_and_add(VectorType &dst, const VectorType &src) const
    {
      transfer->restrict_and_add_block(dst, src, n_rhs);
    }

    void
    prolongate_and_add(VectorType &dst, const VectorType &src) const
    {
      transfer->prolongate_and_add_block(dst, src, n_rhs);
    }

  private:
    ObserverPointer<const MGTransferBase<dim, Number>> transfer;
    const unsigned int                                 n_rhs;
  };

  /**
   * Block counterpart of ProjectedDiagonalPreconditioner: Chat_block =
   * Pi * D^{-1} * Pi applied independently to each of n_rhs blocks. D^{-1}
   * doesn't vary with the RHS block (same operator/diagonal for every
   * column, same principle project_block() and vmult_plain_block() rely
   * on), so rather than materializing an n_rhs-times-replicated diagonal
   * vector, dst.scale(diag) becomes a single Kokkos::parallel_for over
   * n_rhs*stride that indexes the diagonal by i % stride.
   */
  template <typename MatrixAdapterType, typename VectorType>
  class BlockProjectedDiagonalPreconditioner : public EnableObserverPointer
  {
  public:
    using Number = typename VectorType::value_type;

    BlockProjectedDiagonalPreconditioner(
      const MatrixAdapterType                            &matrix_in,
      const std::shared_ptr<DiagonalMatrix<VectorType>> &inverse_diagonal_in,
      const unsigned int                                  n_rhs)
      : matrix(&matrix_in)
      , inverse_diagonal(inverse_diagonal_in)
      , n_rhs(n_rhs)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      AssertDimension(dst.size(), src.size());
      AssertDimension(dst.size() % n_rhs, 0u);
      const unsigned int stride = dst.size() / n_rhs;

      const Number *diag_ptr = inverse_diagonal->get_vector().get_values();
      const Number *src_ptr  = src.get_values();
      Number       *dst_ptr  = dst.get_values();

      Kokkos::parallel_for(
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
          0, static_cast<std::size_t>(n_rhs) * stride),
        KOKKOS_LAMBDA(const std::size_t idx) {
          const unsigned int i = idx % stride;
          dst_ptr[idx]         = src_ptr[idx] * diag_ptr[i];
        });
      Kokkos::fence();

      matrix->project(dst);
    }

  private:
    ObserverPointer<const MatrixAdapterType>            matrix;
    std::shared_ptr<DiagonalMatrix<VectorType>>          inverse_diagonal;
    const unsigned int                                   n_rhs;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
