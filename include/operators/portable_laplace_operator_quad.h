#ifndef portable_laplace_operator_quad_h
#define portable_laplace_operator_quad_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>

#include <memory>


DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {
    // needed for MatrixFreeTools::compute_diagonal()
    template <int dim, int fe_degree, int n_q_points_1d, typename number>
    class LaplaceOperatorQuad
    {
    public:
      DEAL_II_HOST_DEVICE
      LaplaceOperatorQuad()
      {}

      DEAL_II_HOST_DEVICE void
      operator()(Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> *fe_eval,
                 const int                                                         q_point) const;

      static const unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
    };

    template <int dim, int fe_degree, int n_q_points_1d, typename number>
    DEAL_II_HOST_DEVICE void
    LaplaceOperatorQuad<dim, fe_degree, n_q_points_1d, number>::operator()(
      Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> *fe_eval,
      const int                                                         q_point) const
    {
      auto value = fe_eval->get_value(q_point);
      fe_eval->submit_value(value, q_point);
      fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
    }

  } // namespace internal

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif
