#ifndef portable_mg_transfer_base_h
#define portable_mg_transfer_base_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  class MGTransferBase : public EnableObserverPointer
  {
  public:
    ~MGTransferBase() = default;

    virtual void
    prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    virtual void
    restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    /**
     * Block prolongate_and_add()/restrict_and_add(): applies the
     * respective operation independently to each of n_rhs blocks of
     * dst/src, laid out as n_rhs blocks of dof_stride_fine (dst for
     * prolongate, src for restrict) / dof_stride_coarse each -- the same
     * convention bk1_kokkos_kernels_block.h's
     * KokkosProlongationBatchedBlockKernel()/
     * KokkosRestrictionBatchedBlockKernel() use.
     */
    virtual void
    prolongate_and_add_block(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      const unsigned int                                                     n_rhs) const = 0;

    virtual void
    restrict_and_add_block(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      const unsigned int                                                     n_rhs) const = 0;

    virtual void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) = 0;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
