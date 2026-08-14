#ifndef portable_projected_chebyshev_smoother_h
#define portable_projected_chebyshev_smoother_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * Self-contained Chebyshev semi-iterative smoother for a symmetric
   * operator A (here: Ahat = Pi*A*Pi) preconditioned by a symmetric C
   * (here: Chat = Pi*D^{-1}*Pi). Unlike dealii::PreconditionChebyshev,
   * this does not estimate eigenvalues internally (the caller supplies
   * max_eigenvalue, e.g. via estimate_max_eigenvalue() in
   * portable_projected_diagonal_preconditioner.h), does not carry any
   * state between separate vmult()/step() calls, and implements a
   * single fixed (first-kind) Chebyshev polynomial recurrence -- trading
   * dealii::PreconditionChebyshev's generality for something easy to
   * reason about and keep provably consistent with a Pi-projected
   * operator/preconditioner pair.
   *
   * Recurrence (Varga, "Matrix Iterative Analysis"): with
   * theta = (lambda_max+lambda_min)/2, delta = (lambda_max-lambda_min)/2,
   * sigma = theta/delta:
   *   x_1     = x_0 + C^{-1}(b - A x_0) / theta
   *   rho_0   = delta / theta
   *   rho_k   = 1 / (2 sigma - rho_{k-1})
   *   x_{k+1} = x_k + rho_k rho_{k-1} (x_k - x_{k-1})
   *                 + rho_k (2/delta) C^{-1}(b - A x_k)
   * vmult() takes x_0 = 0; step() takes x_0 = the given dst (continuing
   * from wherever the caller left it) -- the usual "fresh start" /
   * "continue" convention a symmetric V-cycle needs from its pre-/post-
   * smoother pair.
   *
   * If Ahat is identically zero at some level (e.g. every dof there
   * belongs to its own size-1 primal constraint group, so Pi projects
   * everything to zero -- see the comment in setup_smoothers() in
   * program.cc), max_eigenvalue is legitimately 0, hence theta = 0.
   * apply() detects this and returns without touching x, which is the
   * mathematically correct no-op (there is nothing to smooth), rather
   * than dividing by theta.
   */
  template <typename MatrixType, typename PreconditionerType, typename VectorType>
  class ProjectedChebyshevSmoother
  {
  public:
    struct AdditionalData
    {
      AdditionalData() = default;

      unsigned int degree          = 5;
      double       max_eigenvalue  = 1.;
      double       smoothing_range = 15.;

      std::shared_ptr<PreconditionerType> preconditioner;
    };

    void
    initialize(const MatrixType &matrix_in, const AdditionalData &data_in = AdditionalData())
    {
      matrix = &matrix_in;
      data   = data_in;

      const double alpha = data.max_eigenvalue / data.smoothing_range;
      delta              = (data.max_eigenvalue - alpha) * 0.5;
      theta              = (data.max_eigenvalue + alpha) * 0.5;

      // Preallocate scratch vectors once, here, rather than on every
      // apply() call: these are device (Kokkos) vectors, and apply() runs
      // twice per level per V-cycle application, many V-cycles per outer
      // solve -- repeated device alloc/dealloc in that hot loop is a real
      // cost, not just noise.
      matrix_in.initialize_dof_vector(r);
      matrix_in.initialize_dof_vector(z);
      matrix_in.initialize_dof_vector(x_prev);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0;
      apply(dst, src, /* zero_out_dst */ true);
    }

    void
    step(VectorType &dst, const VectorType &src) const
    {
      apply(dst, src, /* zero_out_dst */ false);
    }

    const AdditionalData &
    get_additional_data() const
    {
      return data;
    }

    // x_prev[i] = x[i] (save old x for the caller's next iteration);
    // x[i] = x[i] + factor * z[i]. Replaces "x_prev = x; x.add(factor, z);".
    // Public (not private) because it contains a KOKKOS_LAMBDA -- nvcc's
    // extended-lambda rule requires the enclosing function to not have
    // private/protected access within its class.
    static void
    fused_update(VectorType &x, VectorType &x_prev, const VectorType &z, const double factor)
    {
      using Number          = typename VectorType::value_type;
      const unsigned int n = x.locally_owned_size();

      DeviceVector<Number>       x_view(x.get_values(), n);
      DeviceVector<Number>       x_prev_view(x_prev.get_values(), n);
      DeviceVector<const Number> z_view(z.get_values(), n);
      const Number               f = static_cast<Number>(factor);

      Kokkos::parallel_for(
        "chebyshev_fused_update", n, KOKKOS_LAMBDA(const int i) {
          const Number old_x = x_view(i);
          x_prev_view(i)     = old_x;
          x_view(i)          = old_x + f * z_view(i);
        });
      Kokkos::fence();
    }

    // diff[i] = x[i] - x_prev[i] (implicit, never materialized);
    // x_prev[i] = x[i] (save old x); x[i] = x[i] + c1*diff[i] + c2*z[i].
    // Replaces "diff = x; diff -= x_prev; x_prev = x; x.add(c1, diff);
    // x.add(c2, z);" -- 5 kernel launches down to 1, and drops the diff
    // scratch vector entirely. Public for the same nvcc extended-lambda
    // reason as fused_update() above.
    static void
    fused_recurrence_update(VectorType       &x,
                            VectorType       &x_prev,
                            const VectorType &z,
                            const double      c1,
                            const double      c2)
    {
      using Number          = typename VectorType::value_type;
      const unsigned int n = x.locally_owned_size();

      DeviceVector<Number>       x_view(x.get_values(), n);
      DeviceVector<Number>       x_prev_view(x_prev.get_values(), n);
      DeviceVector<const Number> z_view(z.get_values(), n);
      const Number               C1 = static_cast<Number>(c1);
      const Number               C2 = static_cast<Number>(c2);

      Kokkos::parallel_for(
        "chebyshev_fused_recurrence_update", n, KOKKOS_LAMBDA(const int i) {
          const Number old_x     = x_view(i);
          const Number old_xprev = x_prev_view(i);
          x_prev_view(i)         = old_x;
          x_view(i)              = old_x + C1 * (old_x - old_xprev) + C2 * z_view(i);
        });
      Kokkos::fence();
    }

  private:
    void
    apply(VectorType &x, const VectorType &b, const bool zero_out_dst) const
    {
      if (data.degree == 0 || theta <= 0.)
        return;

      // k = 0: first-order correction, no (x_k - x_{k-1}) history yet.
      // When called from vmult() (zero_out_dst == true), x is identically
      // zero on entry, so matrix->vmult(r, x) is a wasted operator
      // application that always produces r = 0 -- skip it and read the
      // preconditioner straight off b, matching
      // dealii::PreconditionChebyshev's iteration_index == 0 fast path
      // (internal::PreconditionChebyshevImplementation::vector_updates()
      // in precondition.h). step() (post-smoothing) still needs the real
      // matvec, since x there is whatever the caller already has.
      if (zero_out_dst)
        data.preconditioner->vmult(z, b);
      else
        {
          matrix->vmult(r, x);
          r.sadd(-1.0, 1.0, b);
          data.preconditioner->vmult(z, r);
        }

      // x_prev = x; x += (1/theta) z -- one fused kernel instead of a copy
      // plus an axpy (2 separate dealii::Vector calls / kernel launches).
      fused_update(x, x_prev, z, 1.0 / theta);

      if (data.degree < 2 || std::abs(delta) < 1e-40)
        return;

      double       rho   = delta / theta;
      const double sigma = theta / delta;

      for (unsigned int k = 1; k < data.degree; ++k)
        {
          const double rho_new = 1.0 / (2.0 * sigma - rho);

          matrix->vmult(r, x);
          r.sadd(-1.0, 1.0, b);
          data.preconditioner->vmult(z, r);

          // Was 5 separate dealii::Vector calls (diff = x; diff -= x_prev;
          // x_prev = x; x.add(c1, diff); x.add(c2, z)) -- each its own
          // kernel launch. diff is only ever x - x_prev consumed
          // immediately, so it doesn't need to be materialized as a
          // standalone vector at all: one fused kernel reads x[i] and
          // x_prev[i] once each, computes the diff in a register, and
          // writes both x_prev (= old x, for next iteration) and the new x
          // in the same pass.
          fused_recurrence_update(x, x_prev, z, rho_new * rho, rho_new * 2.0 / delta);

          rho = rho_new;
        }
    }

    ObserverPointer<const MatrixType> matrix;
    AdditionalData                    data;
    double                            theta = 0.;
    double                            delta = 0.;

    // Scratch vectors reused across apply() calls -- sized once in
    // initialize(). apply() is neither reentrant nor thread-safe as a
    // result, matching SubdomainVCycleMultigrid's own solution/defect/t
    // members: a given instance corresponds to one V-cycle level, and the
    // caller only ever invokes vmult()/step() on it sequentially, never
    // concurrently or recursively onto itself.
    mutable VectorType r, z, x_prev;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
