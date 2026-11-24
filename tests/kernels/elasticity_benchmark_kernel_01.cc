// Benchmark::LinearElasticityKernel (kernels/elasticity_benchmark_kernel.h)
// against the element matrices
//   lambda (div v, div u) + 2 mu (eps(v), eps(u))
// of FESystem(FE_Q(nm - 1), 3) assembled with FEValues on three deformed
// cells, QGauss(nq), for element-local vectors u: nq = nm + 1 as in the
// benchmark, and nq = nm, nm + 2. Two elements per batch, so the last batch
// is padded, and one or two teams, so a team goes through several batches.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include <random>

#include "../tests.h"

#include "kernels/elasticity_benchmark_kernel.h"

using namespace dealii;

constexpr double lambda = 1.7;
constexpr double mu     = 0.6;

template <unsigned int nm, unsigned int nq>
void
test(const unsigned int n_teams)
{
  using Kernel                    = Benchmark::LinearElasticityKernel<double, nm, nq>;
  constexpr unsigned int dim      = 3;
  constexpr unsigned int degree   = nm - 1;
  constexpr unsigned int n_nodes  = nm * nm * nm;
  constexpr unsigned int n_points = nq * nq * nq;

  deallog << "degree = " << degree << ", n_q_points_1d = " << nq << ", teams = " << n_teams
          << std::endl;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria, {3u, 1u, 1u}, Point<dim>(), Point<dim>(3, 1, 1));
  GridTools::transform(
    [](const Point<dim> &p)
    {
      Point<dim> result = p;
      result[0] += 0.2 * p[1] * p[2];
      result[1] += 0.1 * std::sin(p[0]) + 0.3 * p[2];
      result[2] *= 1. + 0.2 * p[0];
      return result;
    },
    tria);

  const FE_Q<dim>       fe_q(degree);
  const FESystem<dim>   fe(fe_q, dim);
  const QGauss<dim>     quadrature(nq);
  const MappingQ<dim>   mapping(1);
  FEValues<dim>         fe_values(mapping,
                          fe,
                          quadrature,
                          update_gradients | update_JxW_values | update_inverse_jacobians);
  const unsigned int    n_cells    = tria.n_active_cells();
  const unsigned int    n_dofs     = fe.n_dofs_per_cell();
  std::vector<unsigned> hierarchic = FETools::lexicographic_to_hierarchic_numbering<dim>(degree);

  // 1D data of FE_Q and of the collocation basis on the Gauss points
  const internal::MatrixFreeFunctions::ShapeInfo<double> shape_info(QGauss<1>{nq}, fe_q);
  const auto                                            &univariate = shape_info.data.front();
  const std::vector<double> basis(univariate.shape_values.begin(), univariate.shape_values.end());
  const std::vector<double> dbasis(univariate.shape_gradients_collocation.begin(),
                                   univariate.shape_gradients_collocation.end());

  const Kernel          kernel(n_cells, 2, basis, dbasis, lambda, mu);
  const std::size_t     n_padded = kernel.n_elements_padded();
  typename Kernel::View d_G("G", n_padded * 10 * n_points), d_in("in", n_padded * 3 * n_nodes),
    d_out("out", n_padded * 3 * n_nodes);
  auto G   = Kokkos::create_mirror_view(d_G);
  auto in  = Kokkos::create_mirror_view(d_in);
  auto out = Kokkos::create_mirror_view(d_out);

  // kernel point (a, b, c) and node (i, j, k): x along a and i
  const auto point = [](const unsigned int a, const unsigned int b, const unsigned int c)
  { return a + nq * (b + nq * c); };
  const auto lexicographic = [](const unsigned int i, const unsigned int j, const unsigned int k)
  { return i + nm * (j + nm * k); };

  std::mt19937                           gen(7);
  std::uniform_real_distribution<double> dist(-1., 1.);

  std::vector<Vector<double>> u(n_cells, Vector<double>(n_dofs)), reference(n_cells);
  unsigned int                e = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values.reinit(cell);

      for (unsigned int a = 0; a < nq; ++a)
        for (unsigned int b = 0; b < nq; ++b)
          for (unsigned int c = 0; c < nq; ++c)
            {
              const unsigned int q    = point(a, b, c);
              const std::size_t  base = e * 10 * n_points + (a * nq + b) * nq + c;
              G(base)                 = fe_values.JxW(q);
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int x = 0; x < dim; ++x)
                  G(base + (1 + 3 * d + x) * n_points) = fe_values.inverse_jacobian(q)[d][x];
            }

      for (unsigned int comp = 0; comp < dim; ++comp)
        for (unsigned int i = 0; i < nm; ++i)
          for (unsigned int j = 0; j < nm; ++j)
            for (unsigned int k = 0; k < nm; ++k)
              {
                const double value                                                  = dist(gen);
                in(comp * n_padded * n_nodes + e * n_nodes + (i * nm + j) * nm + k) = value;
                u[e](fe.component_to_system_index(comp, hierarchic[lexicographic(i, j, k)])) =
                  value;
              }

      const FEValuesExtractors::Vector displacement(0);
      FullMatrix<double>               matrix(n_dofs, n_dofs);
      for (unsigned int q = 0; q < n_points; ++q)
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            matrix(i, j) += (lambda * fe_values[displacement].divergence(i, q) *
                               fe_values[displacement].divergence(j, q) +
                             2. * mu *
                               scalar_product(fe_values[displacement].symmetric_gradient(i, q),
                                              fe_values[displacement].symmetric_gradient(j, q))) *
                            fe_values.JxW(q);
      reference[e].reinit(n_dofs);
      matrix.vmult(reference[e], u[e]);
      ++e;
    }

  Kokkos::deep_copy(d_G, G);
  Kokkos::deep_copy(d_in, in);
  kernel.apply(d_G, d_in, d_out, n_teams);
  Kokkos::fence();
  Kokkos::deep_copy(out, d_out);

  double error = 0., norm = 0.;
  for (unsigned int e = 0; e < n_cells; ++e)
    for (unsigned int comp = 0; comp < dim; ++comp)
      for (unsigned int i = 0; i < nm; ++i)
        for (unsigned int j = 0; j < nm; ++j)
          for (unsigned int k = 0; k < nm; ++k)
            {
              const double expected = reference[e](
                fe.component_to_system_index(comp, hierarchic[lexicographic(i, j, k)]));
              error = std::max(error,
                               std::abs(out(comp * n_padded * n_nodes + e * n_nodes +
                                            (i * nm + j) * nm + k) -
                                        expected));
              norm  = std::max(norm, std::abs(expected));
            }

  if (error < 1e-12 * norm)
    deallog << "OK" << std::endl;
  else
    deallog << "relative error " << error / norm << std::endl;
}



int
main(int argc, char *argv[])
{
  Kokkos::ScopeGuard kokkos(argc, argv);
  initlog();

  for (const unsigned int n_teams : {1u, 2u})
    {
      // the benchmark's nq = nm + 1
      test<2, 3>(n_teams);
      test<3, 4>(n_teams);
      test<4, 5>(n_teams);
      // nq = nm and nq = nm + 2
      test<3, 3>(n_teams);
      test<2, 4>(n_teams);
      test<3, 5>(n_teams);
    }
}
