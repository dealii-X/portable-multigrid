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

    // the derivative of stress() at grad_u in the direction grad_du
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    linearized_stress(const Tensor<2, dim, Number> &, const Tensor<2, dim, Number> &grad_du) const
    {
      return stress(grad_du);
    }
  };



  // Saint Venant-Kirchhoff hyperelasticity: the first Piola-Kirchhoff stress
  // P = F S with F = I + grad u, E = (F^T F - I) / 2 and
  // S = lambda tr(E) I + 2 mu E. The operator is then the nonlinear residual
  // R(u), not a linear map.
  template <int dim, typename Number>
  struct SaintVenantKirchhoffMaterial
  {
    Number lambda = 1.;
    Number mu     = 1.;

    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    stress(const Tensor<2, dim, Number> &grad_u) const
    {
      const Tensor<2, dim, Number> F = deformation_gradient(grad_u);
      const Tensor<2, dim, Number> S = second_piola_kirchhoff(F);
      return multiply(F, S);
    }

    // the derivative of stress() at grad_u in the direction G = grad_du:
    // dP = G S + F dS with dS = lambda tr(dE) I + 2 mu dE, dE = sym(F^T G)
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    linearized_stress(const Tensor<2, dim, Number> &grad_u,
                      const Tensor<2, dim, Number> &grad_du) const
    {
      const Tensor<2, dim, Number> F = deformation_gradient(grad_u);
      const Tensor<2, dim, Number> S = second_piola_kirchhoff(F);

      // dS = mu (F^T G + G^T F) + lambda tr(F^T G) I
      Tensor<2, dim, Number> FtG;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += F[k][i] * grad_du[k][j];
            FtG[i][j] = sum;
          }
      Number trace_dE = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        trace_dE += FtG[d][d];
      Tensor<2, dim, Number> dS;
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            dS[i][j] = mu * (FtG[i][j] + FtG[j][i]);
          dS[i][i] += lambda * trace_dE;
        }

      const Tensor<2, dim, Number> GS = multiply(grad_du, S);
      Tensor<2, dim, Number>       dP = multiply(F, dS);
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          dP[i][j] += GS[i][j];
      return dP;
    }

  private:
    DEAL_II_HOST_DEVICE static Tensor<2, dim, Number>
    multiply(const Tensor<2, dim, Number> &A, const Tensor<2, dim, Number> &B)
    {
      Tensor<2, dim, Number> AB;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += A[i][k] * B[k][j];
            AB[i][j] = sum;
          }
      return AB;
    }

    DEAL_II_HOST_DEVICE static Tensor<2, dim, Number>
    deformation_gradient(const Tensor<2, dim, Number> &grad_u)
    {
      Tensor<2, dim, Number> F = grad_u;
      for (unsigned int d = 0; d < dim; ++d)
        F[d][d] += Number(1.);
      return F;
    }

    // S = lambda tr(E) I + 2 mu E with E = (F^T F - I) / 2
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    second_piola_kirchhoff(const Tensor<2, dim, Number> &F) const
    {
      // S = 2 E = F^T F - I first
      Tensor<2, dim, Number> S;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += F[k][i] * F[k][j];
            S[i][j] = sum;
          }
      Number trace_E = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        {
          S[d][d] -= Number(1.);
          trace_E += Number(0.5) * S[d][d];
        }
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            S[i][j] *= mu;
          S[i][i] += lambda * trace_E;
        }
      return S;
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

    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    linearized_stress(const Tensor<2, dim, Number> &, const Tensor<2, dim, Number> &grad_du) const
    {
      return grad_du;
    }
  };



  // Cell operator (grad v, material.stress(grad u)). FEEvaluationType is
  // Portable::FEEvaluation (one cell per team) or its batched counterpart
  // (a batch of cells per team); the stress need not be symmetric. fused:
  // the stress is computed inside the batched evaluate_and_apply().
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            typename Number,
            typename Material,
            typename FEEvaluationType,
            bool fused = false>
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

      if constexpr (fused)
        fe_eval.evaluate_and_apply(EvaluationFlags::gradients,
                                   [&](const Tensor<2, dim, Number> &grad_u, const int)
                                   { return material.stress(grad_u); });
      else
        {
          fe_eval.evaluate(EvaluationFlags::gradients);
          data->for_each_quad_point(
            [&](const int &q_point)
            { fe_eval.submit_gradient(material.stress(fe_eval.get_gradient(q_point)), q_point); });
        }

      fe_eval.integrate(EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }

  private:
    Material material;
  };



  // grad u at all quadrature points of all colors, point q of cell c at
  // values(n_q_points c + q, i, j); LayoutLeft: the point index runs fastest
  template <int dim, typename Number>
  struct QuadraturePointGradients
  {
    using ViewType =
      Kokkos::View<Number *[dim][dim], Kokkos::LayoutLeft, MemorySpace::Default::kokkos_space>;

    ViewType values;

    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    get(const std::size_t point) const
    {
      Tensor<2, dim, Number> grad;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          grad[i][j] = values(point, i, j);
      return grad;
    }

    DEAL_II_HOST_DEVICE void
    set(const std::size_t point, const Tensor<2, dim, Number> &grad) const
    {
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          values(point, i, j) = grad[i][j];
    }
  };



  // linearize = false: the residual (grad v, material.stress(grad u)), storing
  // grad u in gradients; true: the linearized operator
  // (grad v, material.linearized_stress(grad u, grad du)) with grad u read
  // from gradients. FEEvaluationType is the batched FEEvaluation.
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            typename Number,
            typename Material,
            typename FEEvaluationType,
            bool fused,
            bool linearize>
  class LocalElasticityLinearizationOperator
  {
  public:
    static constexpr unsigned int n_local_dofs = dim * Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);

    DEAL_II_HOST_DEVICE
    LocalElasticityLinearizationOperator(const Material                              &material,
                                         const QuadraturePointGradients<dim, Number> &gradients)
      : material(material)
      , gradients(gradients)
    {}

    DEAL_II_HOST_DEVICE void
    operator()(const typename FEEvaluationType::data_type *data,
               const DeviceVector<Number>                 &src,
               DeviceVector<Number>                       &dst) const
    {
      FEEvaluationType fe_eval(data);

      fe_eval.read_dof_values(src);

      // q_point runs over the points of all cells of the batch
      const auto point_operation = [&](const Tensor<2, dim, Number> &grad, const int q_point)
      {
        const std::size_t point =
          static_cast<std::size_t>(fe_eval.get_current_cell_index(q_point)) * n_q_points +
          q_point % n_q_points;
        if (linearize)
          {
            return material.linearized_stress(gradients.get(point), grad);
          }
        else
          {
            gradients.set(point, grad);
            return material.stress(grad);
          }
      };

      if constexpr (fused)
        {
          fe_eval.evaluate_and_apply(EvaluationFlags::gradients, point_operation);
        }
      else
        {
          fe_eval.evaluate(EvaluationFlags::gradients);
          data->for_each_quad_point(
            [&](const int &q_point) {
              fe_eval.submit_gradient(point_operation(fe_eval.get_gradient(q_point), q_point),
                                      q_point);
            });
        }

      fe_eval.integrate(EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }

  private:
    Material                              material;
    QuadraturePointGradients<dim, Number> gradients;
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
