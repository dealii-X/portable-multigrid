#ifndef portable_local_elasticity_operator_h
#define portable_local_elasticity_operator_h

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Isotropic linear elasticity: stress = lambda tr(eps) I + 2 mu eps with
  // eps = sym(grad u).
  template <int dim, typename Number>
  struct LinearElasticMaterial
  {
    Number lambda = 1.;
    Number mu     = 1.;

    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    stress(const Tensor<2, dim, Number> &grad_u) const
    {
      Number divergence = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        divergence += grad_u[d][d];

      Tensor<2, dim, Number> result;
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            result[i][j] = mu * (grad_u[i][j] + grad_u[j][i]);
          result[i][i] += lambda * divergence;
        }
      return result;
    }
  };



  // stress = grad u, the vector Laplacian; for tests.
  template <int dim, typename Number>
  struct IdentityMaterial
  {
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    stress(const Tensor<2, dim, Number> &grad_u) const
    {
      return grad_u;
    }
  };



  // Cell operator (grad v, material.stress(grad u)). FEEvaluationType is
  // Portable::FEEvaluation (one cell per team) or its batched counterpart
  // (a batch of cells per team); the stress need not be symmetric.
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            typename Number,
            typename Material,
            typename FEEvaluationType>
  class LocalElasticityOperator
  {
  public:
    static constexpr unsigned int n_local_dofs = dim * Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);

    DEAL_II_HOST_DEVICE
    LocalElasticityOperator(const Material &material)
      : material(material)
    {}

    DEAL_II_HOST_DEVICE void
    operator()(const typename FEEvaluationType::data_type *data,
               const DeviceVector<Number>                 &src,
               DeviceVector<Number>                       &dst) const
    {
      FEEvaluationType fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      data->for_each_quad_point(
        [&](const int &q_point)
        { fe_eval.submit_gradient(material.stress(fe_eval.get_gradient(q_point)), q_point); });

      fe_eval.integrate(EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }

  private:
    Material material;
  };



  // The same form at one quadrature point, for MatrixFreeTools::compute_diagonal().
  template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Material>
  class ElasticityOperatorQuad
  {
  public:
    static constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

    DEAL_II_HOST_DEVICE
    ElasticityOperatorQuad(const Material &material)
      : material(material)
    {}

    DEAL_II_HOST_DEVICE void
    operator()(Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> *fe_eval,
               const int                                                           q_point) const
    {
      fe_eval->submit_gradient(material.stress(fe_eval->get_gradient(q_point)), q_point);
    }

  private:
    Material material;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
