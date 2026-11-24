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

// Test Copy::Portable::internal::batched::cell_loop() with the operator
// (v, a u) + (grad v, grad u) and a variable coefficient a, for the
// automatic batch size and for requested ones, see
// batched_fe_evaluation_common.h for the reference.

#include "batched_fe_evaluation_common.h"


struct Helmholtz
{
  static constexpr int n_components = 1;

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
    double         value = 0.;
    Tensor<1, dim> gradient;
    for (unsigned int j = 0; j < u.size(); ++j)
      {
        value += u(j) * fe_values.shape_value(j, q);
        gradient += u(j) * fe_values.shape_grad(j, q);
      }

    for (unsigned int i = 0; i < u.size(); ++i)
      local_dst(i) += (coefficient * value * fe_values.shape_value(i, q) +
                       gradient * fe_values.shape_grad(i, q)) *
                      fe_values.JxW(q);
  }
};



template <int dim, int fe_degree, int n_q_points_1d>
void
run_cell_loop(const bool lobatto = false)
{
  for (const Mesh mesh : {Mesh::deformed, Mesh::hanging_nodes, Mesh::sheared})
    for (const bool use_coloring : {false, true})
      for (const int batch_size : {-1, 1, 5})
        test<dim, fe_degree, n_q_points_1d, Helmholtz>(
          lobatto, mesh, use_coloring, batch_size, true);
}



int
main()
{
  initlog();
  Kokkos::initialize();

  run_cell_loop<2, 1, 2>();
  run_cell_loop<2, 3, 4>();
  run_cell_loop<3, 1, 2>();
  run_cell_loop<3, 2, 3>();
  run_cell_loop<3, 2, 3>(true);

  Kokkos::finalize();
  return 0;
}
