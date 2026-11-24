// Copy::Portable::internal::batched::FEEvaluation::evaluate_and_apply() with
// the collocation, the transform-to-collocation and the general kernel, see
// batched_fe_evaluation_common.h:
//  - (grad v, a grad u) for a scalar element,
//  - (grad v, a grad u + 0.5 (grad u)^T) for dim components, a flux that
//    couples the components and is not symmetric.

#include "batched_fe_evaluation_common.h"


struct ScalarLaplace
{
  static constexpr int  n_components = 1;
  static constexpr bool fused        = true;

  template <typename Gradient>
  DEAL_II_HOST_DEVICE static Gradient
  flux(const Gradient &gradient, const double coefficient)
  {
    return coefficient * gradient;
  }

  template <int dim>
  static void
  reference(const FEValues<dim>  &fe_values,
            const unsigned int    q,
            const double          coefficient,
            const Vector<double> &u,
            Vector<double>       &local_dst)
  {
    Tensor<1, dim> gradient;
    for (unsigned int j = 0; j < u.size(); ++j)
      gradient += u(j) * fe_values.shape_grad(j, q);

    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) += coefficient * (gradient * fe_values.shape_grad(i, q)) * fe_values.JxW(q);
  }
};



template <int dim>
struct CoupledFlux
{
  static constexpr int  n_components = dim;
  static constexpr bool fused        = true;

  DEAL_II_HOST_DEVICE static Tensor<2, dim>
  flux(const Tensor<2, dim> &gradient, const double coefficient)
  {
    Tensor<2, dim> result;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        result[i][j] = coefficient * gradient[i][j] + 0.5 * gradient[j][i];
    return result;
  }

  static void
  reference(const FEValues<dim>  &fe_values,
            const unsigned int    q,
            const double          coefficient,
            const Vector<double> &u,
            Vector<double>       &local_dst)
  {
    const FEValuesExtractors::Vector vector(0);

    Tensor<2, dim> gradient;
    for (unsigned int j = 0; j < u.size(); ++j)
      gradient += u(j) * fe_values[vector].gradient(j, q);

    const Tensor<2, dim> f = coefficient * gradient + 0.5 * transpose(gradient);
    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) += scalar_product(f, fe_values[vector].gradient(i, q)) * fe_values.JxW(q);
  }
};



int
main()
{
  initlog();
  Kokkos::initialize();

  deallog.push("scalar");
  // transformation to collocation
  run<2, 2, 3, ScalarLaplace>();
  run<3, 2, 3, ScalarLaplace>();
  run<3, 2, 4, ScalarLaplace>();
  // general kernel
  run<3, 1, 3, ScalarLaplace>();
  // collocation
  run<2, 2, 3, ScalarLaplace>(true);
  run<3, 2, 3, ScalarLaplace>(true);
  deallog.pop();

  deallog.push("coupled");
  run<2, 2, 3, CoupledFlux<2>>();
  run<3, 2, 3, CoupledFlux<3>>();
  run<3, 2, 4, CoupledFlux<3>>();
  run<2, 1, 3, CoupledFlux<2>>();
  run<3, 1, 3, CoupledFlux<3>>();
  run<2, 2, 3, CoupledFlux<2>>(true);
  run<3, 2, 3, CoupledFlux<3>>(true);
  deallog.pop();

  Kokkos::finalize();
  return 0;
}
