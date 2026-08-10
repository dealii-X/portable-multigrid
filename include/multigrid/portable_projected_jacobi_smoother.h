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
   * Damped Jacobi smoother whose elementary correction is sandwiched as
   * Chat = Pi * (omega * D^{-1}) * Pi, where Pi is MatrixType::project()
   * (the homogeneous primal-constraint projector) and D^{-1} is the
   * diagonal preconditioner of MatrixType. Chat is symmetric by
   * construction (Pi and D^{-1} are each symmetric) independent of
   * whether Pi and D^{-1} commute, so this smoother is a valid symmetric
   * pre-/post-smoother pair (vmult = start from zero, step = continue)
   * for a V-cycle whose level operator is itself Pi-projected
   * (MatrixType::vmult() == Pi * A * Pi).
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
      VectorType residual, correction;
      residual.reinit(dst, true);
      correction.reinit(dst, true);

      const VectorType &inverse_diagonal = data.preconditioner->get_vector();

      for (unsigned int it = 0; it < data.n_iterations; ++it)
        {
          // residual = src - A * dst  (A == matrix->vmult(), already Pi * A * Pi)
          matrix->vmult(residual, dst);
          residual.sadd(-1.0, 1.0, src);

          // sandwich Pi on both sides of the diagonal scaling
          matrix->project(residual);

          correction = residual;
          correction.scale(inverse_diagonal);
          correction *= data.omega;

          matrix->project(correction);

          dst += correction;
        }
    }

  private:
    ObserverPointer<const MatrixType> matrix;
    AdditionalData                    data;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
