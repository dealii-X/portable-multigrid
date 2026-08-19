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

    // Alternate-kernel-implementation counterparts of prolongate_and_add()/
    // restrict_and_add(), mirroring LaplaceOperatorBase::vmult_new()'s role
    // (portable_laplace_operator_base.h) -- meant for A/B correctness/
    // performance comparison against an alternate kernel implementation of
    // the exact same transfer, not a functional alternative. Only
    // Portable::PolynomialTransfer implements these for real
    // (BK1::Parallel::KokkosProlongationBatchedKernelAbstracted()/
    // KokkosRestrictionBatchedKernelAbstracted(), kernels/bk1_kokkos_kernels.h);
    // GeometricTransferCore/ContinuousTransfer have no such alternate kernel
    // and throw DEAL_II_NOT_IMPLEMENTED() -- callers (e.g.
    // tests/poisson_cube_bk3/program.cc's prolong_restrict_comparison_timing())
    // must only call these on levels known to be a PolynomialTransfer.
    virtual void
    prolongate_and_add_new(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    virtual void
    restrict_and_add_new(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    virtual void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) = 0;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
