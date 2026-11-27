// Portable::symmetric_eigenbasis (Jacobi) and
// Portable::symmetric_eigenbasis_closed_form
// (kernels/portable_symmetric_eigenbasis.h) evaluated in a device kernel, against deal.II's
// eigenvectors() (QL with implicit shifts) on the host:
// - eigenvalues
// - spectral projectors sum_{a in group} N_a (x) N_a over groups of equal
//   eigenvalues, which do not depend on the basis chosen in a repeated
//   eigenspace
// - orthonormality and the reconstruction A = sum_a lambda_a N_a (x) N_a
// Matrices: C = F^T F of random F, repeated and nearly repeated eigenvalues
// (C = I, uniaxial stretch, splittings 1e-8 and 1e-13), unsorted diagonal,
// very different scales and a strongly graded matrix.

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <Kokkos_Core.hpp>

#include <random>
#include <string>
#include <vector>

#include "../tests.h"

#include "kernels/portable_symmetric_eigenbasis.h"

using namespace dealii;

template <int dim>
Tensor<2, dim>
identity()
{
  return unit_symmetric_tensor<dim>();
}

template <int dim>
Tensor<2, dim>
rotation(const double angle)
{
  Tensor<2, dim> R;
  if constexpr (dim == 2)
    {
      R[0][0] = R[1][1] = std::cos(angle);
      R[0][1]           = -std::sin(angle);
      R[1][0]           = std::sin(angle);
    }
  else
    {
      // rotation about the axis (1, 2, 3) / sqrt(14)
      const Tensor<1, dim> n = Tensor<1, dim>({1., 2., 3.}) / std::sqrt(14.);
      Tensor<2, dim>       K;
      K[0][1] = -n[2], K[0][2] = n[1], K[1][0] = n[2];
      K[1][2] = -n[0], K[2][0] = -n[1], K[2][1] = n[0];
      R = identity<dim>() + std::sin(angle) * K + (1. - std::cos(angle)) * K * K;
    }
  return R;
}

template <int dim>
Tensor<2, dim>
rotated_diagonal(const std::array<double, dim> &d, const double angle)
{
  Tensor<2, dim> D;
  for (unsigned int i = 0; i < dim; ++i)
    D[i][i] = d[i];
  const Tensor<2, dim> R = rotation<dim>(angle);
  return R * D * transpose(R);
}

template <Portable::EigenSolver eigen_solver, int dim>
DEAL_II_HOST_DEVICE Portable::SymmetricEigenbasis<dim, double>
eigenbasis(const Tensor<2, dim> &T)
{
  if constexpr (eigen_solver == Portable::EigenSolver::closed_form)
    {
      return Portable::symmetric_eigenbasis_closed_form(T);
    }
  else
    {
      return Portable::symmetric_eigenbasis(T);
    }
}

template <int dim, Portable::EigenSolver eigen_solver>
void
test()
{
  deallog << "dim = " << dim << ", "
          << (eigen_solver == Portable::EigenSolver::closed_form ? "closed form" : "jacobi")
          << std::endl;

  std::vector<std::pair<std::string, Tensor<2, dim>>> cases;
  // the reference is computed for cases[c] / factor[c]: deal.II's QL
  // fails at these scales
  std::vector<double> factor;
  cases.emplace_back("identity", identity<dim>());
  {
    std::array<double, dim> d;
    d.fill(1.);
    d[0] = 2.25;
    cases.emplace_back("uniaxial stretch, rotated", rotated_diagonal<dim>(d, 0.7));
    for (const auto &[split, name] : {std::pair{1e-8, "1e-8"}, std::pair{1e-13, "1e-13"}})
      {
        d.fill(3.);
        d[0] = 1.;
        d[1] = 1. + split;
        cases.emplace_back(std::string("splitting ") + name, rotated_diagonal<dim>(d, 0.4));
      }
  }
  {
    Tensor<2, dim> D;
    for (unsigned int i = 0; i < dim; ++i)
      D[i][i] = (i == 0) ? 3. : (i == 1 ? 1. : 2.);
    cases.emplace_back("unsorted diagonal", D);
  }
  {
    Tensor<2, dim> G;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        G[i][j] = (i == j) ? std::pow(1e10, i) : 1e-3 * (i + j + 1);
    cases.emplace_back("graded", G);
  }

  std::mt19937                           gen(11);
  std::uniform_real_distribution<double> dist(-0.4, 0.4);
  for (unsigned int k = 0; k < 20; ++k)
    {
      Tensor<2, dim> F;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          F[i][j] = (i == j) + dist(gen);
      factor.resize(cases.size(), 1.);
      factor.push_back((k == 0) ? 1e-150 : (k == 1 ? 1e150 : 1.));
      cases.emplace_back("C = F^T F, random F " + std::to_string(k),
                         factor.back() * transpose(F) * F);
    }

  const unsigned int n = cases.size();
  factor.resize(n, 1.);
  using MemorySpace = MemorySpace::Default::kokkos_space;
  Kokkos::View<double **, MemorySpace> d_A("A", n, dim * dim), d_values("values", n, dim),
    d_vectors("vectors", n, dim * dim);
  auto A = Kokkos::create_mirror_view(d_A);
  for (unsigned int c = 0; c < n; ++c)
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        A(c, i * dim + j) = cases[c].second[i][j];
  Kokkos::deep_copy(d_A, A);

  Kokkos::parallel_for(
    "symmetric_eigenbasis",
    Kokkos::RangePolicy<MemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const int c) {
      Tensor<2, dim> T;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          T[i][j] = d_A(c, i * dim + j);
      const auto eigen = eigenbasis<eigen_solver>(T);
      for (unsigned int a = 0; a < dim; ++a)
        {
          d_values(c, a) = eigen.values[a];
          for (unsigned int d = 0; d < dim; ++d)
            d_vectors(c, a * dim + d) = eigen.vectors[a][d];
        }
    });
  Kokkos::fence();
  auto values  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_values);
  auto vectors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_vectors);

  for (unsigned int c = 0; c < n; ++c)
    {
      const Tensor<2, dim> &T         = cases[c].second;
      const double          scale     = T.norm();
      const Tensor<2, dim>  T_sym     = symmetrize(T);
      auto                  reference = eigenvectors(symmetrize(T / factor[c]));
      for (auto &[value, vector] : reference)
        value *= factor[c];

      Tensor<1, dim> lambda;
      Tensor<2, dim> N;
      for (unsigned int a = 0; a < dim; ++a)
        {
          lambda[a] = values(c, a);
          for (unsigned int d = 0; d < dim; ++d)
            N[a][d] = vectors(c, a * dim + d);
        }

      double value_error = 0.;
      for (unsigned int a = 0; a < dim; ++a)
        value_error = std::max(value_error, std::abs(lambda[a] - reference[a].first));

      Tensor<2, dim> reconstruction, NNt;
      for (unsigned int a = 0; a < dim; ++a)
        reconstruction += lambda[a] * outer_product(N[a], N[a]);
      NNt                               = N * transpose(N);
      const double reconstruction_error = (reconstruction - T_sym).norm();
      const double orthonormality_error = (NNt - identity<dim>()).norm();

      // groups of equal reference eigenvalues; the projector error is
      // ~ eps scale / gap, so a group closes at a relative gap of 1e-6
      double       projector_error = 0.;
      unsigned int first           = 0;
      while (first < dim)
        {
          unsigned int last = first + 1;
          while (last < dim && reference[last - 1].first - reference[last].first < 1e-6 * scale)
            ++last;
          Tensor<2, dim> P, P_ref;
          for (unsigned int a = first; a < last; ++a)
            {
              P += outer_product(N[a], N[a]);
              P_ref += outer_product(reference[a].second, reference[a].second);
            }
          projector_error = std::max(projector_error, (P - P_ref).norm());
          first           = last;
        }

      const bool ok = value_error < 1e-14 * scale && reconstruction_error < 1e-14 * scale &&
                      orthonormality_error < 1e-14 && projector_error < 1e-9;
      deallog << cases[c].first << ": ";
      if (ok)
        deallog << "OK" << std::endl;
      else
        deallog << "eigenvalue " << value_error / scale << ", reconstruction "
                << reconstruction_error / scale << ", orthonormality " << orthonormality_error
                << ", projector " << projector_error << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  Kokkos::ScopeGuard kokkos(argc, argv);
  initlog();

  test<2, Portable::EigenSolver::jacobi>();
  test<3, Portable::EigenSolver::jacobi>();
  test<2, Portable::EigenSolver::closed_form>();
  test<3, Portable::EigenSolver::closed_form>();
}
