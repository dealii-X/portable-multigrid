#ifndef portable_polynomial_transfer_base_h
#define portable_polynomial_transfer_base_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  class PolynomialTransferBase : public EnableObserverPointer
  {
  public:
    ~PolynomialTransferBase() = default;

    virtual void
    prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) = 0;
  };



} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
