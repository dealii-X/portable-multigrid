// Portable::OgdenMaterial::stress() evaluated in a device kernel against
// references on the host, in 3D and 2D plane strain (F_33 = 1):
// - (alpha1, alpha2) = (2, -2) is Mooney-Rivlin,
//   Psi = c1 (I1_bar - 3) + c2 (I2_bar - 3) + Psi_vol, c1 = mu1 / 2,
//   c2 = -mu2 / 2, with the closed form in invariants
//   S = 2 c1 J^(-2/3) (I - I1 / 3 C^-1)
//     + 2 c2 J^(-4/3) (I1 I - C - 2 / 3 I2 C^-1) + kappa / 2 (J^2 - 1) C^-1
// - any alpha: P = dPsi/dF by fourth-order central differences of Psi(F),
//   Psi from the eigenvalues of C of deal.II's QL algorithm
// Deformations with simple, double and triple principal stretches, and NaN
// for J < 0.

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>

#include <Kokkos_Core.hpp>

#include <random>
#include <string>
#include <vector>

#include "../tests.h"

#include "kernels/portable_local_elasticity_operator.h"

using namespace dealii;

using Material = Portable::OgdenMaterial<3, double>;

Tensor<2, 3>
embed(const Tensor<2, 3> &grad_u)
{
  return grad_u;
}

Tensor<2, 3>
embed(const Tensor<2, 2> &grad_u)
{
  Tensor<2, 3> G;
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      G[i][j] = grad_u[i][j];
  return G;
}

Tensor<2, 3>
identity()
{
  return unit_symmetric_tensor<3>();
}

// first Piola-Kirchhoff stress of Mooney-Rivlin
Tensor<2, 3>
mooney_rivlin(const Material &m, const Tensor<2, 3> &grad_u)
{
  const Tensor<2, 3> I     = identity();
  const Tensor<2, 3> F     = I + grad_u;
  const Tensor<2, 3> C     = transpose(F) * F;
  const Tensor<2, 3> C_inv = invert(C);
  const double       J     = determinant(F);
  const double       I1    = trace(C);
  const double       I2    = 0.5 * (I1 * I1 - trace(C * C));
  const double       c1 = 0.5 * m.mu1, c2 = -0.5 * m.mu2;

  const Tensor<2, 3> S = 2. * c1 * std::pow(J, -2. / 3.) * (I - I1 / 3. * C_inv) +
                         2. * c2 * std::pow(J, -4. / 3.) * (I1 * I - C - 2. / 3. * I2 * C_inv) +
                         0.5 * m.kappa * (J * J - 1.) * C_inv;
  return F * S;
}

double
energy(const Material &m, const Tensor<2, 3> &F)
{
  const double J     = determinant(F);
  const auto   eigen = eigenvectors(symmetrize(transpose(F) * F));
  double       psi   = 0.25 * m.kappa * (J * J - 1. - 2. * std::log(J));
  for (const auto &[lambda_sq, N] : eigen)
    {
      const double lambda_bar_sq = std::pow(J, -2. / 3.) * lambda_sq;
      psi += m.mu1 / m.alpha1 * (std::pow(lambda_bar_sq, 0.5 * m.alpha1) - 1.) +
             m.mu2 / m.alpha2 * (std::pow(lambda_bar_sq, 0.5 * m.alpha2) - 1.);
    }
  return psi;
}

// dPsi/dF_ij for i, j < dim (plane strain: F_33 = 1 fixed)
template <int dim>
Tensor<2, 3>
energy_derivative(const Material &m, const Tensor<2, 3> &grad_u)
{
  const double h = 1e-3;
  Tensor<2, 3> P;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        const auto psi = [&](const double t)
        {
          Tensor<2, 3> F = identity() + grad_u;
          F[i][j] += t;
          return energy(m, F);
        };
        P[i][j] = (-psi(2. * h) + 8. * psi(h) - 8. * psi(-h) + psi(-2. * h)) / (12. * h);
      }
  return P;
}

template <int dim, Portable::EigenSolver eigen_solver>
std::vector<double>
device_stress(const Material &m, const std::vector<Tensor<2, dim>> &grads)
{
  const Portable::OgdenMaterial<dim, double, eigen_solver> material{
    m.mu1, m.alpha1, m.mu2, m.alpha2, m.kappa};

  const unsigned int n = grads.size();
  using MemorySpace    = MemorySpace::Default::kokkos_space;
  Kokkos::View<double **, MemorySpace> d_grad("grad", n, dim * dim), d_P("P", n, dim * dim);
  auto                                 grad = Kokkos::create_mirror_view(d_grad);
  for (unsigned int c = 0; c < n; ++c)
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        grad(c, i * dim + j) = grads[c][i][j];
  Kokkos::deep_copy(d_grad, grad);

  Kokkos::parallel_for(
    "ogden_stress",
    Kokkos::RangePolicy<MemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const int c) {
      Tensor<2, dim, double> G;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          G[i][j] = d_grad(c, i * dim + j);
      const Tensor<2, dim, double> P = material.stress(G);
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          d_P(c, i * dim + j) = P[i][j];
    });
  Kokkos::fence();
  auto P = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_P);
  return std::vector<double>(P.data(), P.data() + n * dim * dim);
}

template <int dim, Portable::EigenSolver eigen_solver>
void
test()
{
  deallog << "dim = " << dim << ", eigen solver "
          << (eigen_solver == Portable::EigenSolver::closed_form ? "closed form" : "jacobi")
          << std::endl;

  std::vector<std::pair<std::string, Tensor<2, dim>>> cases;
  const auto diagonal = [](const std::array<double, dim> &d)
  {
    Tensor<2, dim> G;
    for (unsigned int i = 0; i < dim; ++i)
      G[i][i] = d[i];
    return G;
  };
  {
    std::array<double, dim> d;
    d.fill(0.);
    cases.emplace_back("u = 0", Tensor<2, dim>());
    d[0] = 0.5;
    cases.emplace_back("uniaxial stretch", diagonal(d));
    d[0] = -0.4;
    cases.emplace_back("uniaxial compression", diagonal(d));
    d[0] = 0.3;
    d[1] = 1e-10;
    cases.emplace_back("nearly uniaxial", diagonal(d));
    d.fill(0.1);
    cases.emplace_back("dilatation", diagonal(d));
    Tensor<2, dim> G;
    G[0][1] = 0.4;
    cases.emplace_back("simple shear", G);
  }
  std::mt19937                           gen(5);
  std::uniform_real_distribution<double> dist(-0.25, 0.25);
  for (unsigned int k = 0; k < 10; ++k)
    {
      Tensor<2, dim> G;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          G[i][j] = dist(gen);
      cases.emplace_back("random " + std::to_string(k), G);
    }
  {
    // F = diag(-1, 1, 1): J = -1
    Tensor<2, dim> G;
    G[0][0] = -2.;
    cases.emplace_back("J < 0", G);
  }

  std::vector<Tensor<2, dim>> grads;
  for (const auto &c : cases)
    grads.push_back(c.second);

  const std::vector<std::pair<std::string, Material>> materials = {
    {"Mooney-Rivlin", {0.8, 2., -0.3, -2., 2.5}}, {"Ogden", {0.9, 3.7, -0.2, -1.3, 2.5}}};

  for (const auto &[name, m] : materials)
    {
      const std::vector<double> P_device = device_stress<dim, eigen_solver>(m, grads);
      for (unsigned int c = 0; c < cases.size(); ++c)
        {
          Tensor<2, 3> P;
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              P[i][j] = P_device[(c * dim + i) * dim + j];

          deallog << name << ", " << cases[c].first << ": ";
          if (cases[c].first == "J < 0")
            {
              bool all_nan = true;
              for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                  all_nan = all_nan && std::isnan(P[i][j]);
              deallog << (all_nan ? "NaN" : "not NaN") << std::endl;
              continue;
            }

          const Tensor<2, 3> grad_u = embed(cases[c].second);
          // stress scale: the shear modulus
          const double scale    = 0.5 * (m.mu1 * m.alpha1 + m.mu2 * m.alpha2);
          double       error_fd = 0., error_mr = 0.;
          const auto   P_fd = energy_derivative<dim>(m, grad_u);
          const auto   P_mr = mooney_rivlin(m, grad_u);
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              {
                error_fd = std::max(error_fd, std::abs(P[i][j] - P_fd[i][j]) / scale);
                error_mr = std::max(error_mr, std::abs(P[i][j] - P_mr[i][j]) / scale);
              }

          const bool is_mooney_rivlin = (m.alpha1 == 2. && m.alpha2 == -2.);
          if (error_fd < 1e-9 && (!is_mooney_rivlin || error_mr < 1e-13))
            deallog << "OK" << std::endl;
          else
            deallog << "finite differences " << error_fd
                    << (is_mooney_rivlin ? ", Mooney-Rivlin " + std::to_string(error_mr) : "")
                    << std::endl;
        }
    }
}



int
main(int argc, char *argv[])
{
  Kokkos::ScopeGuard kokkos(argc, argv);
  initlog();

  test<2, Portable::EigenSolver::closed_form>();
  test<3, Portable::EigenSolver::closed_form>();
  test<2, Portable::EigenSolver::jacobi>();
  test<3, Portable::EigenSolver::jacobi>();
}
