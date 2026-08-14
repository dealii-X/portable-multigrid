#ifndef portable_dd_preconditioner_base_h
#define portable_dd_preconditioner_base_h

#include <deal.II/lac/la_parallel_vector.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * Common interface for the domain-decomposition preconditioners (BNN,
   * BDDC) SolverProjectedCG::solve_dd() can drive, letting it balance the
   * initial residual once, before the outer CG loop starts, without
   * needing to know which concrete preconditioner it's driving.
   */
  template <int dim, typename Number>
  class DDPreconditionerBase
  {
  public:
    virtual ~DDPreconditionerBase() = default;

    virtual void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const = 0;
    /**
     * Projects r in place onto whatever "balanced residual" subspace this
     * preconditioner's vmult() relies on for its cheaper action (a no-op
     * for preconditioners, e.g. BDDC, with no such notion). Called once by
     * solve_dd(), right after the initial residual r_0 = b - A*x_0 is
     * computed -- not per iteration, since balancedness then propagates to
     * every later r_k automatically under a matching vmult().
     */
    virtual void
    project_initial_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &r) const = 0;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
