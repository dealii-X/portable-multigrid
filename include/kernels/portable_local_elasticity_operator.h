#ifndef portable_local_elasticity_operator_h
#define portable_local_elasticity_operator_h

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <limits>

#include "kernels/portable_symmetric_eigenbasis.h"

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



  // Decoupled Ogden hyperelasticity (docs: lung_application ogden_material):
  // Psi = sum_a sum_p mu_p / alpha_p (lambda_bar_a^alpha_p - 1)
  //       + kappa / 4 (J^2 - 1 - 2 ln J), lambda_bar_a = J^(-1/3) lambda_a,
  // P = F S. In 2D plane strain: lambda_3 = 1. NaN for J <= 0.
  // eigen_solver: the eigenbasis of C by symmetric_eigenbasis_closed_form()
  // or symmetric_eigenbasis() (Jacobi)
  template <int dim, typename Number, EigenSolver eigen_solver = EigenSolver::closed_form>
  struct OgdenMaterial
  {
    Number mu1    = 1.;
    Number alpha1 = 2.;
    Number mu2    = 0.;
    Number alpha2 = -2.;
    Number kappa  = 1.;

    // S = sum_a s_a N_a (x) N_a, s_a = S_iso_a + p_vol / lambda_a^2
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    stress(const Tensor<2, dim, Number> &grad_u) const
    {
      const Principal pr = principal(grad_u);
      if (!pr.valid)
        return nan_tensor();

      Tensor<2, dim, Number> P;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            Number sum = 0.;
            for (unsigned int a = 0; a < dim; ++a)
              sum += (pr.S_iso[a] + pr.p_vol / pr.eigen.values[a]) * pr.FN[i][a] *
                     pr.eigen.vectors[a][j];
            P[i][j] = sum;
          }
      return P;
    }

    // dP = G S + F DS, G = grad_du, DS of eqs. (11), (12) in the eigenbasis:
    // DS = sum_ab D_ab N_a (x) N_b with M_ab = N_a . DC N_b, DC = F^T G + G^T F
    DEAL_II_HOST_DEVICE Tensor<2, dim, Number>
    linearized_stress(const Tensor<2, dim, Number> &grad_u,
                      const Tensor<2, dim, Number> &grad_du) const
    {
      const Principal pr = principal(grad_u);
      if (!pr.valid)
        return nan_tensor();
      const Tensor<1, dim, Number> &lambda_sq = pr.eigen.values;
      const Tensor<2, dim, Number> &N         = pr.eigen.vectors;

      Number GN[dim][dim];
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int a = 0; a < dim; ++a)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += grad_du[i][k] * N[a][k];
            GN[i][a] = sum;
          }
      Number M[dim][dim];
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          {
            Number sum = 0.;
            for (unsigned int i = 0; i < dim; ++i)
              sum += pr.FN[i][a] * GN[i][b] + GN[i][a] * pr.FN[i][b];
            M[a][b] = sum;
          }

      // c1_ab of eq. (7)
      Number c1[dim][dim];
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          {
            c1[a][b] = ((a == b ? pr.gamma[a] : Number(0.)) -
                        (pr.gamma[a] + pr.gamma[b]) / Number(3.) + pr.gamma_sum / Number(9.)) /
                       (lambda_sq[a] * lambda_sq[b]);
            if (a == b)
              c1[a][b] -= Number(2.) * pr.S_iso[a] / lambda_sq[a];
          }

      // DJ / J = C^-1 : DC / 2
      Number DJ_over_J = 0.;
      for (unsigned int a = 0; a < dim; ++a)
        DJ_over_J += Number(0.5) * M[a][a] / lambda_sq[a];

      // volumetric part: kappa J^2 DJ/J C^-1 + p_vol DC^-1, DC^-1 = -C^-1 DC C^-1
      const Number cbrt_eps = Kokkos::cbrt(std::numeric_limits<Number>::epsilon());
      Number       D[dim][dim];
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          {
            Number value = -pr.p_vol * M[a][b] / (lambda_sq[a] * lambda_sq[b]);
            if (a == b)
              {
                for (unsigned int c = 0; c < dim; ++c)
                  value += Number(0.5) * c1[a][c] * M[c][c];
                value += kappa * pr.J * pr.J * DJ_over_J / lambda_sq[a];
              }
            else
              {
                // c2_ab of eq. (9). S_iso carries an error ~ eps mu, so the
                // quotient has an error ~ eps mu / |lambda_b^2 - lambda_a^2|;
                // the symmetric limit (c1_aa + c1_bb - 2 c1_ab) / 4 one of
                // O(|lambda_b^2 - lambda_a^2|^2): switch at eps^(1/3)
                const Number difference = lambda_sq[b] - lambda_sq[a];
                const Number c2         = (Kokkos::abs(difference) <
                                   cbrt_eps * Number(0.5) * (lambda_sq[a] + lambda_sq[b])) ?
                                            Number(0.25) * (c1[a][a] + c1[b][b]) - Number(0.5) * c1[a][b] :
                                            (pr.S_iso[b] - pr.S_iso[a]) / difference;
                value += c2 * M[a][b];
              }
            D[a][b] = value;
          }

      // dP = sum_b (s_b G N_b + sum_a D_ab F N_a) (x) N_b
      Tensor<2, dim, Number> dP;
      for (unsigned int i = 0; i < dim; ++i)
        {
          Number row[dim];
          for (unsigned int b = 0; b < dim; ++b)
            {
              Number sum = (pr.S_iso[b] + pr.p_vol / lambda_sq[b]) * GN[i][b];
              for (unsigned int a = 0; a < dim; ++a)
                sum += D[a][b] * pr.FN[i][a];
              row[b] = sum;
            }
          for (unsigned int j = 0; j < dim; ++j)
            {
              Number sum = 0.;
              for (unsigned int b = 0; b < dim; ++b)
                sum += row[b] * N[b][j];
              dP[i][j] = sum;
            }
        }
      return dP;
    }

  private:
    struct Principal
    {
      bool                             valid;
      Number                           J;
      SymmetricEigenbasis<dim, Number> eigen;
      // FN[i][a] = (F N_a)_i
      Number FN[dim][dim];
      // S_iso_a of eq. (4), gamma_a of eq. (3) over all three stretches
      Number S_iso[dim];
      Number gamma[3];
      Number gamma_sum;
      // kappa / 2 (J^2 - 1), eq. (5)
      Number p_vol;
    };

    DEAL_II_HOST_DEVICE Principal
    principal(const Tensor<2, dim, Number> &grad_u) const
    {
      Principal pr;

      Tensor<2, dim, Number> F = grad_u;
      for (unsigned int d = 0; d < dim; ++d)
        F[d][d] += Number(1.);
      Tensor<2, dim, Number> C;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += F[k][i] * F[k][j];
            C[i][j] = sum;
          }

      pr.J = determinant(F);
      // also excludes the FE_INVALID of pow(J < 0, -2/3)
      pr.valid = (pr.J > Number(0.));
      if (!pr.valid)
        return pr;

      if constexpr (eigen_solver == EigenSolver::closed_form)
        {
          pr.eigen = symmetric_eigenbasis_closed_form(C);
        }
      else
        {
          pr.eigen = symmetric_eigenbasis(C);
        }
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int a = 0; a < dim; ++a)
          {
            Number sum = 0.;
            for (unsigned int k = 0; k < dim; ++k)
              sum += F[i][k] * pr.eigen.vectors[a][k];
            pr.FN[i][a] = sum;
          }

      // beta_a = sum_p mu_p lambda_bar_a^alpha_p,
      // gamma_a = sum_p mu_p alpha_p lambda_bar_a^alpha_p
      const Number J_pow = Kokkos::pow(pr.J, Number(-2. / 3.));
      Number       beta[3];
      Number       beta_sum = 0.;
      pr.gamma_sum          = 0.;
      for (unsigned int a = 0; a < 3; ++a)
        {
          const Number lambda_bar_sq = J_pow * (a < dim ? pr.eigen.values[a] : Number(1.));
          const Number pow1          = Kokkos::pow(lambda_bar_sq, Number(0.5) * alpha1);
          const Number pow2          = Kokkos::pow(lambda_bar_sq, Number(0.5) * alpha2);
          beta[a]                    = mu1 * pow1 + mu2 * pow2;
          pr.gamma[a]                = mu1 * alpha1 * pow1 + mu2 * alpha2 * pow2;
          beta_sum += beta[a];
          pr.gamma_sum += pr.gamma[a];
        }
      for (unsigned int a = 0; a < dim; ++a)
        pr.S_iso[a] = (beta[a] - beta_sum / Number(3.)) / pr.eigen.values[a];
      pr.p_vol = Number(0.5) * kappa * (pr.J * pr.J - Number(1.));
      return pr;
    }

    DEAL_II_HOST_DEVICE static Tensor<2, dim, Number>
    nan_tensor()
    {
      Tensor<2, dim, Number> T;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          T[i][j] = std::numeric_limits<Number>::quiet_NaN();
      return T;
    }

    DEAL_II_HOST_DEVICE static Number
    determinant(const Tensor<2, dim, Number> &F)
    {
      if constexpr (dim == 2)
        {
          return F[0][0] * F[1][1] - F[0][1] * F[1][0];
        }
      else
        {
          return F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) -
                 F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) +
                 F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
        }
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

  // The linearized form at one quadrature point, at the grad u stored by
  // evaluate_residual(), for MatrixFreeTools::compute_diagonal().
  template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Material>
  class ElasticityLinearizedOperatorQuad
  {
  public:
    static constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

    DEAL_II_HOST_DEVICE
    ElasticityLinearizedOperatorQuad(const Material                              &material,
                                     const QuadraturePointGradients<dim, Number> &gradients)
      : material(material)
      , gradients(gradients)
    {}

    DEAL_II_HOST_DEVICE void
    operator()(Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> *fe_eval,
               const int                                                           q_point) const
    {
      const std::size_t point =
        static_cast<std::size_t>(fe_eval->get_current_cell_index()) * n_q_points + q_point;
      fe_eval->submit_gradient(
        material.linearized_stress(gradients.get(point), fe_eval->get_gradient(q_point)),
        q_point);
    }

  private:
    Material                              material;
    QuadraturePointGradients<dim, Number> gradients;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
