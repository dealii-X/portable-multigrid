// Custom::Parallel::FEEvaluationImplTransformToCollocation
// (matrix_free/portable_evaluation_kernels.h) against dense sums with the
// FEValues shape values and gradients of FE_Q on the unit cell, for a batch
// of two cells with different DoF values:
//   evaluate() and evaluate_values() + evaluate_gradients(): u(x_q), grad u(x_q)
//   evaluate_gradients_and_multiply_symmetric_tensor(): G(x_q) grad u(x_q)
//   integrate() and integrate_gradients<true>() + integrate_values():
//     sum_q (phi_i v_q + grad phi_i . w_q)
// The 1D shape tables come from deal.II's ShapeInfo, as in
// LaplaceOperatorBK3.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include <random>

#include "../tests.h"

#include "matrix_free/portable_evaluation_kernels.h"

using namespace dealii;

using Number    = double;
using View      = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;
using HostTable = std::vector<double>;

constexpr int n_cells = 2;

// phi[q][i] and dphi[q][i][d] at the Gauss points of the unit cell, DoFs in
// lexicographic order (the kernel's numbering)
template <int dim>
void
reference_shape_data(const unsigned int                        fe_degree,
                     const unsigned int                        n_q_points_1d,
                     std::vector<std::vector<double>>         &phi,
                     std::vector<std::vector<Tensor<1, dim>>> &dphi)
{
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  const FE_Q<dim>     fe(fe_degree);
  const MappingQ<dim> mapping(1);
  const QGauss<dim>   quadrature(n_q_points_1d);
  FEValues<dim>       fe_values(mapping, fe, quadrature, update_values | update_gradients);
  fe_values.reinit(triangulation.begin_active());

  const std::vector<unsigned int> hierarchic_of_lexicographic =
    FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

  phi.assign(quadrature.size(), std::vector<double>(fe.n_dofs_per_cell()));
  dphi.assign(quadrature.size(), std::vector<Tensor<1, dim>>(fe.n_dofs_per_cell()));
  for (unsigned int q = 0; q < quadrature.size(); ++q)
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      {
        phi[q][i]  = fe_values.shape_value(hierarchic_of_lexicographic[i], q);
        dphi[q][i] = fe_values.shape_grad(hierarchic_of_lexicographic[i], q);
      }
}



View
to_device(const HostTable &host, const std::string &name)
{
  View view(name, host.size());
  auto mirror = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < host.size(); ++i)
    mirror(i) = host[i];
  Kokkos::deep_copy(view, mirror);
  return view;
}



HostTable
to_host(const View &view)
{
  auto mirror = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(mirror, view);
  return HostTable(mirror.data(), mirror.data() + mirror.size());
}



HostTable
random_table(const std::size_t size, std::mt19937 &gen)
{
  std::uniform_real_distribution<double> dist(-1., 1.);
  HostTable                              table(size);
  for (auto &entry : table)
    entry = dist(gen);
  return table;
}



// max |a - b| relative to max |b| over the first b.size() entries
double
relative_difference(const HostTable &a, const HostTable &b)
{
  double diff = 0., norm = 0.;
  for (std::size_t i = 0; i < b.size(); ++i)
    {
      diff = std::max(diff, std::abs(a[i] - b[i]));
      norm = std::max(norm, std::abs(b[i]));
    }
  return diff / std::max(norm, 1e-300);
}



void
report(const std::string &name, const double err)
{
  if (err < 1e-12)
    deallog << name << ": OK" << std::endl;
  else
    deallog << name << ": rel_err = " << err << std::endl;
}



using Policy = Kokkos::TeamPolicy<>;
using Member = Policy::member_type;

template <int dim, int fe_degree, int n_q_points_1d>
using FEEval =
  Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>;

// Runs `body(fe_eval, team_member)` once on a team of the default execution
// space.
template <int dim, int fe_degree, int n_q_points_1d, typename Body>
void
run_on_team(const View &shape_values, const View &co_shape_gradients, const Body &body)
{
  constexpr bool is_serial =
    std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
  const int team_size = is_serial ? 1 : Utilities::pow(n_q_points_1d, dim - 1) * n_cells;

  Kokkos::parallel_for(
    Policy(1, team_size), KOKKOS_LAMBDA(const Member &team_member) {
      const FEEval<dim, fe_degree, n_q_points_1d> fe_eval(team_member,
                                                          shape_values.data(),
                                                          co_shape_gradients.data(),
                                                          n_cells,
                                                          n_cells,
                                                          team_member.league_rank(),
                                                          team_member.team_rank(),
                                                          team_member.team_size());
      body(fe_eval, team_member);
    });
  Kokkos::fence();
}



template <int dim, int fe_degree, int n_q_points_1d>
void
test()
{
  constexpr int n_dofs     = Utilities::pow(fe_degree + 1, dim);
  constexpr int n_q_points = Utilities::pow(n_q_points_1d, dim);
  constexpr int n_qp_batch = n_cells * n_q_points;
  constexpr int n_sym      = dim * (dim + 1) / 2;

  using FE = FEEval<dim, fe_degree, n_q_points_1d>;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  // 1D tables as LaplaceOperatorBK3 builds them
  const QGauss<1>                                        quadrature_1d(n_q_points_1d);
  const FE_Q<dim>                                        fe(fe_degree);
  const internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(quadrature_1d, fe);
  const auto                                            &univariate = shape_info.data.front();
  const View                                             shape_values =
    to_device(HostTable(univariate.shape_values.begin(), univariate.shape_values.end()),
              "shape_values");
  const View co_shape_gradients =
    to_device(HostTable(univariate.shape_gradients_collocation.begin(),
                        univariate.shape_gradients_collocation.end()),
              "co_shape_gradients");

  std::vector<std::vector<double>>         phi;
  std::vector<std::vector<Tensor<1, dim>>> dphi;
  reference_shape_data<dim>(fe_degree, n_q_points_1d, phi, dphi);

  std::mt19937 gen(42);

  // inputs: DoF values packed per cell, quadrature-point values and
  // gradients ([d][cell][q]), G as [cell][component][q]
  const HostTable u = random_table(n_cells * n_dofs, gen);
  const HostTable v = random_table(n_qp_batch, gen);
  const HostTable w = random_table(dim * n_qp_batch, gen);
  const HostTable G = random_table(n_cells * n_sym * n_q_points, gen);

  // dense references
  HostTable u_q(n_qp_batch), grad_q(dim * n_qp_batch), G_grad_q(dim * n_qp_batch),
    r(n_cells * n_dofs);
  for (int c = 0; c < n_cells; ++c)
    {
      for (int q = 0; q < n_q_points; ++q)
        {
          Tensor<1, dim> grad;
          double         value = 0.;
          for (int i = 0; i < n_dofs; ++i)
            {
              value += u[c * n_dofs + i] * phi[q][i];
              grad += u[c * n_dofs + i] * dphi[q][i];
            }
          u_q[c * n_q_points + q] = value;

          // symmetric G from its packed upper triangle (d1 <= d2)
          Tensor<2, dim> G_q;
          int            component = 0;
          for (int d1 = 0; d1 < dim; ++d1)
            for (int d2 = d1; d2 < dim; ++d2, ++component)
              G_q[d1][d2] = G_q[d2][d1] = G[c * n_sym * n_q_points + component * n_q_points + q];
          const Tensor<1, dim> G_grad = G_q * grad;

          for (int d = 0; d < dim; ++d)
            {
              grad_q[d * n_qp_batch + c * n_q_points + q]   = grad[d];
              G_grad_q[d * n_qp_batch + c * n_q_points + q] = G_grad[d];
            }
        }

      for (int i = 0; i < n_dofs; ++i)
        {
          double sum = 0.;
          for (int q = 0; q < n_q_points; ++q)
            {
              sum += phi[q][i] * v[c * n_q_points + q];
              for (int d = 0; d < dim; ++d)
                sum += dphi[q][i][d] * w[d * n_qp_batch + c * n_q_points + q];
            }
          r[c * n_dofs + i] = sum;
        }
    }

  // the kernels work in place on buffers of n_qp_batch entries per field
  const auto u_buffer = [&]()
  {
    HostTable buffer(n_qp_batch, 0.);
    std::copy(u.begin(), u.end(), buffer.begin());
    return buffer;
  };
  const View d_G = to_device(G, "G");

  // evaluate() with values and gradients
  {
    const View values  = to_device(u_buffer(), "values");
    const View grads   = to_device(HostTable(dim * n_qp_batch), "gradients");
    const View scratch = to_device(HostTable(2 * n_qp_batch), "scratch");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.evaluate(values.data(),
                         grads.data(),
                         scratch.data(),
                         EvaluationFlags::values | EvaluationFlags::gradients);
      });
    report("evaluate: values", relative_difference(to_host(values), u_q));
    report("evaluate: gradients", relative_difference(to_host(grads), grad_q));
  }

  // evaluate_values() + evaluate_gradients<false>() / <true>()
  {
    const View values  = to_device(u_buffer(), "values");
    const View grads   = to_device(HostTable(dim * n_qp_batch), "gradients");
    const View scratch = to_device(HostTable(2 * n_qp_batch), "scratch");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.evaluate_values(values.data(), values.data(), scratch.data());
        team_member.team_barrier();
        fe_eval.evaluate_gradients(values.data(), grads.data());
      });
    report("evaluate_values", relative_difference(to_host(values), u_q));
    report("evaluate_gradients", relative_difference(to_host(grads), grad_q));

    const View grads_add = to_device(w, "gradients_add");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.template evaluate_gradients<true>(values.data(), grads_add.data());
      });
    HostTable expected(dim * n_qp_batch);
    for (int i = 0; i < dim * n_qp_batch; ++i)
      expected[i] = w[i] + grad_q[i];
    report("evaluate_gradients<add>", relative_difference(to_host(grads_add), expected));
  }

  // evaluate_values() + evaluate_gradients_and_multiply_symmetric_tensor()
  {
    const View values  = to_device(u_buffer(), "values");
    const View grads   = to_device(HostTable(dim * n_qp_batch), "gradients");
    const View scratch = to_device(HostTable(2 * n_qp_batch), "scratch");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.evaluate_values(values.data(), values.data(), scratch.data());
        team_member.team_barrier();
        fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(
          d_G, Custom::Parallel::CellRangeIdView(), values.data(), grads.data());
      });
    report("evaluate_gradients_and_multiply_symmetric_tensor",
           relative_difference(to_host(grads), G_grad_q));
  }

  // integrate() with values and gradients
  {
    const View values  = to_device(v, "values");
    const View grads   = to_device(w, "gradients");
    const View scratch = to_device(HostTable(2 * n_qp_batch), "scratch");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.integrate(values.data(),
                          grads.data(),
                          scratch.data(),
                          EvaluationFlags::values | EvaluationFlags::gradients);
      });
    report("integrate", relative_difference(to_host(values), r));
  }

  // integrate_gradients<true>() + integrate_values()
  {
    const View values  = to_device(v, "values");
    const View grads   = to_device(w, "gradients");
    const View scratch = to_device(HostTable(2 * n_qp_batch), "scratch");
    run_on_team<dim, fe_degree, n_q_points_1d>(
      shape_values,
      co_shape_gradients,
      KOKKOS_LAMBDA(const FE &fe_eval, const Member &team_member) {
        fe_eval.template integrate_gradients<true>(grads.data(), values.data());
        team_member.team_barrier();
        fe_eval.integrate_values(values.data(), values.data(), scratch.data());
      });
    report("integrate_gradients<add> + integrate_values", relative_difference(to_host(values), r));
  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  initlog();

  test<1, 1, 2>();
  test<1, 3, 4>();
  test<1, 2, 4>();
  test<2, 1, 2>();
  test<2, 2, 3>();
  test<2, 4, 5>();
  test<2, 2, 4>();
  test<3, 1, 2>();
  test<3, 2, 3>();
  test<3, 3, 4>();
  test<3, 2, 4>();
}
