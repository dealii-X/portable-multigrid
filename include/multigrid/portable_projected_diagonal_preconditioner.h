#ifndef portable_projected_diagonal_preconditioner_h
#define portable_projected_diagonal_preconditioner_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * Preconditioner Chat = Pi * D^{-1} * Pi, where Pi is MatrixType::project()
   * (the homogeneous primal-constraint projector) and D^{-1} is a diagonal
   * scaling. Chat is symmetric by construction (Pi and D^{-1} are each
   * symmetric) regardless of whether Pi and D^{-1} commute, so handing this
   * to PreconditionChebyshev as its PreconditionerType keeps vmult()/step()
   * a valid symmetric pre-/post-smoother pair for a V-cycle whose level
   * operator is itself Pi-projected (MatrixType::vmult() == Pi * A * Pi).
   * Unlike the hand-rolled ProjectedJacobiSmoother, no separate damping
   * factor is needed here: PreconditionChebyshev derives its own optimal
   * scaling from the eigenvalue estimate of Chat-preconditioned A-hat.
   */
  template <typename MatrixType, typename VectorType>
  class ProjectedDiagonalPreconditioner
  {
  public:
    ProjectedDiagonalPreconditioner(
      const MatrixType                                   &matrix_in,
      const std::shared_ptr<DiagonalMatrix<VectorType>> &inverse_diagonal_in)
      : matrix(&matrix_in)
      , inverse_diagonal(inverse_diagonal_in)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = src;
      // matrix->project(dst);
      dst.scale(inverse_diagonal->get_vector());
      matrix->project(dst);
    }

  private:
    ObserverPointer<const MatrixType>           matrix;
    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal;
  };



  /**
   * Estimate the largest eigenvalue of Chat*Ahat (as used e.g. to feed
   * PreconditionChebyshev::AdditionalData::max_eigenvalue) by plain power
   * iteration, mirroring what dealii::internal::power_iteration does
   * internally for PreconditionChebyshev.
   *
   * This exists because deal.II's own default eigenvalue estimator
   * (EigenvalueAlgorithm::lanczos) runs an actual SolverCG solve seeded
   * with a generic, non-V-projected vector. Ahat = Pi*A*Pi is singular
   * outside V = range(Pi) (its kernel is exactly the "constant per
   * constraint entity" directions), so that internal CG solve hits an
   * exact-zero p*Ahat*p denominator and returns NaN. Ahat and Chat both
   * map any input into V on their very first application, so ordinary
   * power iteration -- which only ever applies the operators, never
   * solves a system -- does not blow up regardless of whether the
   * starting vector lies in V. It still needs real high-frequency
   * content in that starting vector to reliably converge to the
   * *largest* eigenvalue, though (a smooth/constant vector is a poor
   * probe for the dominant mode of a Laplacian-type operator) -- see
   * dealii::internal::set_initial_guess() for a suitable choice.
   */
  template <typename MatrixType, typename PreconditionerType, typename VectorType>
  double
  estimate_max_eigenvalue(const MatrixType         &matrix,
                          const PreconditionerType &preconditioner,
                          VectorType               &eigenvector,
                          const unsigned int        n_iterations)
  {
    double eigenvalue_estimate = 0.;

    eigenvector /= eigenvector.l2_norm();

    VectorType v1, v2;
    v1.reinit(eigenvector, true);
    v2.reinit(eigenvector, true);

    for (unsigned int i = 0; i < n_iterations; ++i)
      {
        matrix.vmult(v2, eigenvector);
        preconditioner.vmult(v1, v2);

        eigenvalue_estimate = eigenvector * v1;

        const double v1_norm = v1.l2_norm();
        if (v1_norm == 0.)
          break;
        v1 /= v1_norm;
        eigenvector.swap(v1);
      }

    return std::abs(eigenvalue_estimate);
  }



  struct EigenvalueBounds
  {
    double min_eigenvalue = 0.;
    double max_eigenvalue = 0.;
  };

  /**
   * Estimate [min, safety_factor*max] eigenvalue bounds of Chat-
   * preconditioned Ahat using the same Lanczos-tridiagonalization-from-CG-
   * coefficients approach dealii::PreconditionChebyshev uses internally
   * (SolverCG::connect_eigenvalues_slot() with an IterationNumberControl),
   * but seeded with a probe vector genuinely projected into V = range(Pi)
   * via MatrixType::project() -- not dealii::AffineConstraints::set_zero(),
   * which only zeroes individually-listed dofs and cannot express "subtract
   * the group mean", so it cannot exactly reproduce Pi.
   *
   * This matters because Ahat = Pi*A*Pi is singular outside V (its kernel
   * is exactly the "constant per constraint entity" directions).
   * dealii::PreconditionChebyshev's default estimator seeds its internal CG
   * solve with dealii::internal::set_initial_guess(), which is not confined
   * to V, and that CG solve then hits an exact-zero p*Ahat*p denominator and
   * returns NaN. Once the probe vector genuinely starts in V, every CG
   * iterate stays in V too (same argument as the outer solver in
   * vmult_fine_correction()), so the solve is well-posed and gives the same
   * quality of estimate dealii::PreconditionChebyshev relies on elsewhere.
   *
   * If Ahat is identically zero at this level (e.g. every dof belongs to
   * its own size-1 primal constraint group, so Pi projects everything to
   * zero -- see the comment in setup_smoothers() in program.cc), the
   * projected probe vector is itself exactly zero; both bounds are then
   * returned as 0, which ProjectedChebyshevSmoother treats as "nothing to
   * smooth" (theta <= 0 -> no-op) rather than dividing by zero.
   */
  template <typename MatrixType, typename PreconditionerType, typename VectorType>
  EigenvalueBounds
  estimate_eigenvalue_bounds(const MatrixType         &matrix,
                             const PreconditionerType &preconditioner,
                             VectorType               &probe_vector,
                             const unsigned int        n_cg_iterations,
                             const double              safety_factor = 1.2)
  {
    matrix.project(probe_vector);

    EigenvalueBounds bounds;

    if (probe_vector.l2_norm() == 0.)
      return bounds;

    std::vector<double> eigenvalues;

    IterationNumberControl control(n_cg_iterations, 1e-10, false, false);
    SolverCG<VectorType>   solver(control);
    solver.connect_eigenvalues_slot(
      [&eigenvalues](const std::vector<double> &values) { eigenvalues = values; });

    VectorType x;
    x.reinit(probe_vector);

    solver.solve(matrix, x, probe_vector, preconditioner);

    if (!eigenvalues.empty())
      {
        bounds.min_eigenvalue = eigenvalues.front();
        bounds.max_eigenvalue = safety_factor * eigenvalues.back();
      }

    return bounds;
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
