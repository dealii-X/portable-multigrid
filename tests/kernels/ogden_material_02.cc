// Portable::OgdenMaterial::linearized_stress() evaluated in a device kernel
// against references on the host, in 3D and 2D plane strain (F_33 = 1):
// - (alpha1, alpha2) = (2, -2), Mooney-Rivlin: the exact derivative of the
//   closed form in invariants, with dJ / J = C^-1 : dC / 2, dI1 = tr dC,
//   dI2 = I1 dI1 - C : dC, dC^-1 = -C^-1 dC C^-1
// - any alpha: fourth-order central differences of P = F S, eqs. (4), (5),
//   evaluated with the eigenvectors of deal.II's QL algorithm
// - u = 0: linear elasticity, mu = (mu1 alpha1 + mu2 alpha2) / 2,
//   lambda = kappa - 2 mu / 3
// - C close to repeated eigenvalues, around the switch to the limit of c2_ab
//   at |lambda_b^2 - lambda_a^2| < 100 eps: grad u = t B and
//   grad u = D + t B with D a stretch with a double eigenvalue, against the
//   Mooney-Rivlin derivative

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "../tests.h"

#include "kernels/portable_local_elasticity_operator.h"

using namespace dealii;

using Material = Portable::OgdenMaterial<3, double>;

template <int dim>
Tensor<2, 3>
embed(const Tensor<2, dim> &A)
{
  Tensor<2, 3> B;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      B[i][j] = A[i][j];
  return B;
}

Tensor<2, 3>
identity()
{
  return unit_symmetric_tensor<3>();
}

// Mooney-Rivlin, c1 = mu1 / 2, c2 = -mu2 / 2: P and its derivative in the
// direction G
std::pair<Tensor<2, 3>, Tensor<2, 3>>
mooney_rivlin(const Material &m, const Tensor<2, 3> &grad_u, const Tensor<2, 3> &G)
{
  const Tensor<2, 3> I     = identity();
  const Tensor<2, 3> F     = I + grad_u;
  const Tensor<2, 3> C     = transpose(F) * F;
  const Tensor<2, 3> C_inv = invert(C);
  const double       J     = determinant(F);
  const double       I1    = trace(C);
  const double       I2    = 0.5 * (I1 * I1 - trace(C * C));
  const double       c1 = 0.5 * m.mu1, c2 = -0.5 * m.mu2;
  const double       J23 = std::pow(J, -2. / 3.), J43 = std::pow(J, -4. / 3.);

  const Tensor<2, 3> A1 = I - I1 / 3. * C_inv;
  const Tensor<2, 3> A2 = I1 * I - C - 2. / 3. * I2 * C_inv;
  const Tensor<2, 3> S =
    2. * c1 * J23 * A1 + 2. * c2 * J43 * A2 + 0.5 * m.kappa * (J * J - 1.) * C_inv;

  const Tensor<2, 3> dC        = transpose(F) * G + transpose(G) * F;
  const double       dJ_over_J = 0.5 * scalar_product(C_inv, dC);
  const double       dI1       = trace(dC);
  const double       dI2       = I1 * dI1 - scalar_product(C, dC);
  const Tensor<2, 3> dC_inv    = -C_inv * dC * C_inv;

  const Tensor<2, 3> dS =
    2. * c1 * J23 * (-2. / 3. * dJ_over_J * A1 - dI1 / 3. * C_inv - I1 / 3. * dC_inv) +
    2. * c2 * J43 *
      (-4. / 3. * dJ_over_J * A2 + dI1 * I - dC - 2. / 3. * (dI2 * C_inv + I2 * dC_inv)) +
    m.kappa * J * J * dJ_over_J * C_inv + 0.5 * m.kappa * (J * J - 1.) * dC_inv;

  return {F * S, G * S + F * dS};
}

// P = F S of eqs. (4), (5) with deal.II's eigenvectors
Tensor<2, 3>
spectral_stress(const Material &m, const Tensor<2, 3> &F)
{
  const double J     = determinant(F);
  const auto   eigen = eigenvectors(symmetrize(transpose(F) * F));
  double       beta[3], beta_sum = 0.;
  for (unsigned int a = 0; a < 3; ++a)
    {
      const double lambda_bar_sq = std::pow(J, -2. / 3.) * eigen[a].first;
      beta[a]                    = m.mu1 * std::pow(lambda_bar_sq, 0.5 * m.alpha1) +
                m.mu2 * std::pow(lambda_bar_sq, 0.5 * m.alpha2);
      beta_sum += beta[a];
    }
  Tensor<2, 3> S;
  for (unsigned int a = 0; a < 3; ++a)
    S += (beta[a] - beta_sum / 3. + 0.5 * m.kappa * (J * J - 1.)) / eigen[a].first *
         outer_product(eigen[a].second, eigen[a].second);
  return F * S;
}

Tensor<2, 3>
spectral_derivative(const Material &m, const Tensor<2, 3> &grad_u, const Tensor<2, 3> &G)
{
  const double h = 1e-3;
  const auto   P = [&](const double t) { return spectral_stress(m, identity() + grad_u + t * G); };
  return (-P(2. * h) + 8. * P(h) - 8. * P(-h) + P(-2. * h)) / (12. * h);
}

template <int dim, Portable::EigenSolver eigen_solver>
std::vector<Tensor<2, dim>>
device_linearized(const Material                    &m,
                  const std::vector<Tensor<2, dim>> &grads,
                  const std::vector<Tensor<2, dim>> &directions)
{
  const Portable::OgdenMaterial<dim, double, eigen_solver> material{
    m.mu1, m.alpha1, m.mu2, m.alpha2, m.kappa};

  const unsigned int n = grads.size();
  using MemorySpace    = MemorySpace::Default::kokkos_space;
  Kokkos::View<double **, MemorySpace> d_in("in", n, 2 * dim * dim), d_dP("dP", n, dim * dim);
  auto                                 in = Kokkos::create_mirror_view(d_in);
  for (unsigned int c = 0; c < n; ++c)
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        {
          in(c, i * dim + j)             = grads[c][i][j];
          in(c, dim * dim + i * dim + j) = directions[c][i][j];
        }
  Kokkos::deep_copy(d_in, in);

  Kokkos::parallel_for(
    "ogden_linearized_stress",
    Kokkos::RangePolicy<MemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const int c) {
      Tensor<2, dim, double> grad_u, G;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            grad_u[i][j] = d_in(c, i * dim + j);
            G[i][j]      = d_in(c, dim * dim + i * dim + j);
          }
      const Tensor<2, dim, double> dP = material.linearized_stress(grad_u, G);
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          d_dP(c, i * dim + j) = dP[i][j];
    });
  Kokkos::fence();
  auto                        dP = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_dP);
  std::vector<Tensor<2, dim>> result(n);
  for (unsigned int c = 0; c < n; ++c)
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        result[c][i][j] = dP(c, i * dim + j);
  return result;
}

// max_ij |A_ij - B_ij| over the in-plane entries
template <int dim>
double
difference(const Tensor<2, dim> &A, const Tensor<2, 3> &B)
{
  double result = 0.;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      result = std::max(result, std::abs(A[i][j] - B[i][j]));
  return result;
}

template <int dim>
Tensor<2, dim>
random_tensor(std::mt19937 &gen, const double amplitude)
{
  std::uniform_real_distribution<double> dist(-amplitude, amplitude);
  Tensor<2, dim>                         T;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      T[i][j] = dist(gen);
  return T;
}

template <int dim, Portable::EigenSolver eigen_solver>
void
test()
{
  deallog << "dim = " << dim << ", eigen solver "
          << (eigen_solver == Portable::EigenSolver::closed_form ? "closed form" : "jacobi")
          << std::endl;

  const std::vector<std::pair<std::string, Material>> materials = {
    {"Mooney-Rivlin", {0.8, 2., -0.3, -2., 2.5}}, {"Ogden", {0.9, 3.7, -0.2, -1.3, 2.5}}};

  std::mt19937 gen(3);

  std::vector<std::pair<std::string, Tensor<2, dim>>> cases;
  {
    const auto diagonal = [](const std::array<double, dim> &d)
    {
      Tensor<2, dim> G;
      for (unsigned int i = 0; i < dim; ++i)
        G[i][i] = d[i];
      return G;
    };
    std::array<double, dim> d;
    d.fill(0.);
    cases.emplace_back("u = 0", Tensor<2, dim>());
    d[0] = 0.5;
    cases.emplace_back("uniaxial stretch", diagonal(d));
    d[0] = -0.4;
    cases.emplace_back("uniaxial compression", diagonal(d));
    d.fill(0.1);
    cases.emplace_back("dilatation", diagonal(d));
    Tensor<2, dim> G;
    G[0][1] = 0.4;
    cases.emplace_back("simple shear", G);
    for (unsigned int k = 0; k < 10; ++k)
      cases.emplace_back("random " + std::to_string(k), random_tensor<dim>(gen, 0.25));
  }
  std::vector<Tensor<2, dim>> grads, directions;
  for (const auto &c : cases)
    {
      grads.push_back(c.second);
      directions.push_back(random_tensor<dim>(gen, 1.));
    }

  for (const auto &[name, m] : materials)
    {
      const double mu     = 0.5 * (m.mu1 * m.alpha1 + m.mu2 * m.alpha2);
      const double lambda = m.kappa - 2. / 3. * mu;
      const auto   dP     = device_linearized<dim, eigen_solver>(m, grads, directions);
      for (unsigned int c = 0; c < cases.size(); ++c)
        {
          const Tensor<2, 3> grad_u = embed<dim>(grads[c]);
          const Tensor<2, 3> G      = embed<dim>(directions[c]);
          const double       scale  = mu * G.norm();

          const double error_fd = difference<dim>(dP[c], spectral_derivative(m, grad_u, G)) / scale;
          const bool   is_mooney_rivlin = (m.alpha1 == 2. && m.alpha2 == -2.);
          const double error_mr =
            is_mooney_rivlin ? difference<dim>(dP[c], mooney_rivlin(m, grad_u, G).second) / scale :
                               0.;
          double error_linear = 0.;
          if (cases[c].first == "u = 0")
            {
              const Tensor<2, 3> linear = mu * (G + transpose(G)) + lambda * trace(G) * identity();
              error_linear              = difference<dim>(dP[c], linear) / scale;
            }

          deallog << name << ", " << cases[c].first << ": ";
          if (error_fd < 1e-9 && error_mr < 1e-13 && error_linear < 1e-13)
            deallog << "OK" << std::endl;
          else
            deallog << "finite differences " << error_fd << ", Mooney-Rivlin " << error_mr
                    << ", linear elasticity " << error_linear << std::endl;
        }
    }

  // repeated eigenvalues: C = I + O(t) and a stretch D + t B with a double
  // eigenvalue (3D: lambda_2 = lambda_3, 2D: lambda_1 = lambda_2)
  {
    const Material &m  = materials[0].second;
    const double    mu = 0.5 * (m.mu1 * m.alpha1 + m.mu2 * m.alpha2);
    Tensor<2, dim>  D;
    D[0][0] = 0.5;
    if (dim == 2)
      D[1][1] = 0.5;
    const Tensor<2, dim> B = random_tensor<dim>(gen, 1.);
    const Tensor<2, dim> G = random_tensor<dim>(gen, 1.);

    std::vector<double> ts;
    for (int e = -14; e <= -4; ++e)
      ts.push_back(std::pow(10., e));
    for (const auto &[base_name, base] :
         {std::pair{std::string("t B"), Tensor<2, dim>()}, std::pair{std::string("D + t B"), D}})
      {
        std::vector<Tensor<2, dim>> sweep_grads, sweep_directions(ts.size(), G);
        for (const double t : ts)
          sweep_grads.push_back(base + t * B);
        const auto dP = device_linearized<dim, eigen_solver>(m, sweep_grads, sweep_directions);
        for (unsigned int k = 0; k < ts.size(); ++k)
          {
            const double error =
              difference<dim>(dP[k],
                              mooney_rivlin(m, embed<dim>(sweep_grads[k]), embed<dim>(G)).second) /
              (mu * G.norm());
            deallog << base_name << ", t = " << std::scientific << std::setprecision(0) << ts[k]
                    << std::defaultfloat << std::setprecision(6) << ": ";
            if (error < 1e-10)
              deallog << "OK" << std::endl;
            else
              deallog << error << std::endl;
          }
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
