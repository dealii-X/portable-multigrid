#ifndef portable_projected_jacobi_smoother_h
#define portable_projected_jacobi_smoother_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <memory>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * Damped Jacobi smoother whose elementary correction is Chat =
   * Pi * (omega * D^{-1}), where Pi is MatrixType::project() (the
   * homogeneous primal-constraint projector) and D^{-1} is the diagonal
   * preconditioner of MatrixType. Matches the same convention as
   * ProjectedChebyshevSmoother / ProjectedDiagonalPreconditioner: dst/src
   * are assumed already in V = range(Pi) on entry (the V-cycle maintains
   * this via its own compensating projections after restrict/prolongate),
   * so matrix->vmult() (== Pi * A, output-only projected) and this
   * smoother only need to project the *output* of the diagonal scaling,
   * not its input -- D^{-1} does not preserve V (it does not commute
   * with Pi in general), but A applied to an already-in-V vector plus a
   * difference of two in-V quantities both stay in V for free, so no
   * extra input-side project() is needed before the scaling.
   */
  template <typename MatrixType, typename VectorType>
  class ProjectedJacobiSmoother
  {
  public:
    struct AdditionalData
    {
      AdditionalData() = default;

      unsigned int n_iterations = 5;
      double       omega        = 0.6;

      std::shared_ptr<DiagonalMatrix<VectorType>> preconditioner;
    };

    void
    initialize(const MatrixType &matrix_in, const AdditionalData &data_in = AdditionalData())
    {
      matrix = &matrix_in;
      data   = data_in;

      // Preallocated once here rather than on every step() call -- see
      // ProjectedChebyshevSmoother's r/z/x_prev/diff for the same reasoning:
      // these are device vectors and step() runs in the V-cycle's hot loop.
      matrix_in.initialize_dof_vector(residual);
      matrix_in.initialize_dof_vector(correction);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0;
      step(dst, src);
    }

    void
    step(VectorType &dst, const VectorType &src) const
    {
      const VectorType &inverse_diagonal = data.preconditioner->get_vector();

      for (unsigned int it = 0; it < data.n_iterations; ++it)
        {
          // residual = src - A * dst  (A == matrix->vmult(), == Pi * A here);
          // already in V since dst and src both are (see class comment).
          matrix->vmult(residual, dst);
          residual.sadd(-1.0, 1.0, src);

          correction = residual;
          correction.scale(inverse_diagonal);
          correction *= data.omega;

          // Necessary: D^{-1} scaling does not preserve V.
          matrix->project(correction);

          dst += correction;
        }
    }

  private:
    ObserverPointer<const MatrixType> matrix;
    AdditionalData                    data;

    mutable VectorType residual, correction;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
