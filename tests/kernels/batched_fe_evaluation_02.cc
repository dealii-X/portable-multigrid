// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// Test Copy::Portable::internal::batched::FEEvaluation for a vector-valued element
// with dim components, see batched_fe_evaluation_common.h. With a variable
// coefficient a, the operators are
//  - (v, a u) + (grad v, grad u),
//  - (v, u) + (eps(v), a eps(u)) with the symmetric gradient eps,
//  - (v, u) + (div v, a div u).

#include "batched_fe_evaluation_common.h"


struct VectorLaplace
{
  template <typename FEEval>
  DEAL_II_HOST_DEVICE static void
  apply(FEEval &fe_eval, const int q_point, const double coefficient)
  {
    fe_eval.submit_value(coefficient * fe_eval.get_value(q_point), q_point);
    fe_eval.submit_gradient(fe_eval.get_gradient(q_point), q_point);
  }

  template <int dim>
  static void
  reference(const FEValues<dim>  &fe_values,
            const unsigned int    q,
            const double          coefficient,
            const Vector<double> &u,
            Vector<double>       &local_dst)
  {
    const FEValuesExtractors::Vector vector(0);

    Tensor<1, dim> value;
    Tensor<2, dim> gradient;
    for (unsigned int j = 0; j < u.size(); ++j)
      {
        value += u(j) * fe_values[vector].value(j, q);
        gradient += u(j) * fe_values[vector].gradient(j, q);
      }

    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) +=
        (coefficient * value * fe_values[vector].value(i, q) +
         scalar_product(gradient, fe_values[vector].gradient(i, q))) *
        fe_values.JxW(q);
  }
};



struct SymmetricGradient
{
  template <typename FEEval>
  DEAL_II_HOST_DEVICE static void
  apply(FEEval &fe_eval, const int q_point, const double coefficient)
  {
    fe_eval.submit_value(fe_eval.get_value(q_point), q_point);
    fe_eval.submit_symmetric_gradient(coefficient *
                                        fe_eval.get_symmetric_gradient(q_point),
                                      q_point);
  }

  template <int dim>
  static void
  reference(const FEValues<dim>  &fe_values,
            const unsigned int    q,
            const double          coefficient,
            const Vector<double> &u,
            Vector<double>       &local_dst)
  {
    const FEValuesExtractors::Vector vector(0);

    Tensor<1, dim>          value;
    SymmetricTensor<2, dim> symmetric_gradient;
    for (unsigned int j = 0; j < u.size(); ++j)
      {
        value += u(j) * fe_values[vector].value(j, q);
        symmetric_gradient += u(j) * fe_values[vector].symmetric_gradient(j, q);
      }

    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) +=
        (value * fe_values[vector].value(i, q) +
         coefficient *
           (symmetric_gradient * fe_values[vector].symmetric_gradient(i, q))) *
        fe_values.JxW(q);
  }
};



struct Divergence
{
  template <typename FEEval>
  DEAL_II_HOST_DEVICE static void
  apply(FEEval &fe_eval, const int q_point, const double coefficient)
  {
    fe_eval.submit_value(fe_eval.get_value(q_point), q_point);
    fe_eval.submit_divergence(coefficient * fe_eval.get_divergence(q_point),
                              q_point);
  }

  template <int dim>
  static void
  reference(const FEValues<dim>  &fe_values,
            const unsigned int    q,
            const double          coefficient,
            const Vector<double> &u,
            Vector<double>       &local_dst)
  {
    const FEValuesExtractors::Vector vector(0);

    Tensor<1, dim> value;
    double         divergence = 0.;
    for (unsigned int j = 0; j < u.size(); ++j)
      {
        value += u(j) * fe_values[vector].value(j, q);
        divergence += u(j) * fe_values[vector].divergence(j, q);
      }

    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) +=
        (value * fe_values[vector].value(i, q) +
         coefficient * divergence * fe_values[vector].divergence(i, q)) *
        fe_values.JxW(q);
  }
};



// The operators above with dim components
template <int dim, typename Operator>
struct WithComponents : Operator
{
  static constexpr int n_components = dim;
};



template <int dim, int fe_degree, int n_q_points_1d>
void
run_all()
{
  deallog.push("gradient");
  run<dim, fe_degree, n_q_points_1d, WithComponents<dim, VectorLaplace>>();
  deallog.pop();

  deallog.push("symmetric_gradient");
  run<dim, fe_degree, n_q_points_1d, WithComponents<dim, SymmetricGradient>>();
  deallog.pop();

  deallog.push("divergence");
  run<dim, fe_degree, n_q_points_1d, WithComponents<dim, Divergence>>();
  deallog.pop();
}



int
main()
{
  initlog();
  Kokkos::initialize();

  run_all<2, 2, 3>();
  run_all<3, 1, 2>();

  Kokkos::finalize();
  return 0;
}
