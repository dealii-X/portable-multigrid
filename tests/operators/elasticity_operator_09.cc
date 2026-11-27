// Portable::ElasticityOperator with the Ogden material linearized at u:
// vmult_linearized() and vmult_linearized_fused(), after evaluate_residual()
// or evaluate_residual_fused(), on the curved, sheared mesh of
// deformed_mesh_util.h. For (alpha1, alpha2) = (2, -2), Mooney-Rivlin,
// against the CPU FEValues reference with the derivative of the closed form
// in invariants (2D: plane strain, C_33 = 1):
//   S = mu1 J^(-2/3) A1 - mu2 J^(-4/3) A2 + kappa / 2 (J^2 - 1) C^-1,
//   A1 = I - I1 / 3 C^-1, A2 = I1 I - C - 2 / 3 I2 C^-1,
//   dJ / J = C^-1 : dC / 2, dI1 = tr dC, dI2 = I1 dI1 - C : dC,
//   dC^-1 = -C^-1 dC C^-1.
// For Mooney-Rivlin and a general Ogden material:
// - the error of the central difference (R(u + eps du) - R(u - eps du)) / (2
//   eps) is O(eps^2): halving eps divides it by 4
// - the linearized operator is symmetric
// - at u = 0 it is linear elasticity with mu = (mu1 alpha1 + mu2 alpha2) / 2
//   and lambda = kappa - 2 mu / 3

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_elasticity_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim>
Tensor<2, dim>
mooney_rivlin_linearized(const double          mu1,
                         const double          mu2,
                         const double          kappa,
                         const Tensor<2, dim> &grad_u,
                         const Tensor<2, dim> &grad_du)
{
  Tensor<2, 3> F, G;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        F[i][j] = grad_u[i][j];
        G[i][j] = grad_du[i][j];
      }
  for (unsigned int d = 0; d < 3; ++d)
    F[d][d] += 1.;
  const Tensor<2, 3> I     = unit_symmetric_tensor<3>();
  const Tensor<2, 3> C     = transpose(F) * F;
  const Tensor<2, 3> C_inv = invert(C);
  const double       J     = determinant(F);
  const double       I1    = trace(C);
  const double       I2    = 0.5 * (I1 * I1 - trace(C * C));
  const double       J23 = std::pow(J, -2. / 3.), J43 = std::pow(J, -4. / 3.);

  const Tensor<2, 3> A1 = I - I1 / 3. * C_inv;
  const Tensor<2, 3> A2 = I1 * I - C - 2. / 3. * I2 * C_inv;
  const Tensor<2, 3> S  = mu1 * J23 * A1 - mu2 * J43 * A2 + 0.5 * kappa * (J * J - 1.) * C_inv;

  const Tensor<2, 3> dC        = transpose(F) * G + transpose(G) * F;
  const double       dJ_over_J = 0.5 * scalar_product(C_inv, dC);
  const double       dI1       = trace(dC);
  const double       dI2       = I1 * dI1 - scalar_product(C, dC);
  const Tensor<2, 3> dC_inv    = -C_inv * dC * C_inv;
  const Tensor<2, 3> dS =
    mu1 * J23 * (-2. / 3. * dJ_over_J * A1 - dI1 / 3. * C_inv - I1 / 3. * dC_inv) -
    mu2 * J43 * (-4. / 3. * dJ_over_J * A2 + dI1 * I - dC - 2. / 3. * (dI2 * C_inv + I2 * dC_inv)) +
    kappa * J * J * dJ_over_J * C_inv + 0.5 * kappa * (J * J - 1.) * dC_inv;
  const Tensor<2, 3> dP = G * S + F * dS;

  Tensor<2, dim> result;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      result[i][j] = dP[i][j];
  return result;
}

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test(const std::string &name, const Portable::OgdenMaterial<dim, double> &material)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::OgdenMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  constexpr double tolerance = 1e-12;

  const bool is_mooney_rivlin = (material.alpha1 == 2. && material.alpha2 == -2.);

  deallog << name << ", dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FESystem<dim> fe(FE_Q<dim>{fe_degree}, dim);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints);
  constraints.close();

  Operator op(dof_handler, constraints, material, false);

  VectorType u, du, v, dst, dst_ref, diag;
  for (VectorType *vec : {&u, &du, &v, &dst, &dst_ref, &diag})
    op.initialize_dof_vector(*vec);

  // v as u, du another pattern, all zero on the constrained DoFs; u smaller
  // than in elasticity_operator_06: J > 0 at all points
  fill_source(u, locally_owned_dofs, &constraints);
  v = u;
  u *= 0.01;
  {
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = constraints.is_constrained(i) ? 0. : std::cos(0.29 * static_cast<double>(i + 2));
    du.import_elements(rw, VectorOperation::insert);
  }

  const auto check = [&](const std::string &name, auto &&apply, const VectorType &src)
  {
    dst = 0.;
    apply(dst, src);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);
  };

  const auto linearized = [&](VectorType &d, const VectorType &s) { op.vmult_linearized(d, s); };
  const auto linearized_fused = [&](VectorType &d, const VectorType &s)
  { op.vmult_linearized_fused(d, s); };

  {
    VectorType residual(u);
    op.evaluate_residual(residual, u);
  }
  if (is_mooney_rivlin)
    {
      reference_hyperelastic_linearized<dim, Number>(
        mapping,
        dof_handler,
        constraints,
        n_q_points_1d,
        [&](const Tensor<2, dim> &grad_u, const Tensor<2, dim> &G) {
          return mooney_rivlin_linearized<dim>(
            material.mu1, material.mu2, material.kappa, grad_u, G);
        },
        locally_owned_dofs,
        locally_relevant_dofs,
        MPI_COMM_WORLD,
        u,
        du,
        dst_ref);
      check("vmult_linearized", linearized, du);
      check("vmult_linearized_fused", linearized_fused, du);
      {
        VectorType residual(u);
        op.evaluate_residual_fused(residual, u);
      }
      check("vmult_linearized after evaluate_residual_fused", linearized, du);
      check("vmult_linearized_fused after evaluate_residual_fused", linearized_fused, du);
    }

  // central differences of R = vmult() against vmult_linearized()
  {
    VectorType tangent(du);
    op.vmult_linearized(tangent, du);

    const auto difference_error = [&](const double eps)
    {
      VectorType u_plus(u), u_minus(u), r_plus(u), r_minus(u);
      u_plus.add(eps, du);
      u_minus.add(-eps, du);
      op.vmult(r_plus, u_plus);
      op.vmult(r_minus, u_minus);
      r_plus -= r_minus;
      r_plus *= 1. / (2. * eps);
      return relative_l2_difference(r_plus, tangent);
    };

    const double ratio = difference_error(2e-3) / difference_error(1e-3);
    deallog << "central difference error ratio eps / (eps / 2): " << std::fixed
            << std::setprecision(2) << ratio << std::defaultfloat << std::endl;
  }

  // v^T A du = du^T A v
  {
    VectorType A_du(du), A_v(du);
    op.vmult_linearized(A_du, du);
    op.vmult_linearized(A_v, v);
    const double v_A_du = v * A_du;
    report("symmetry", std::abs(v_A_du - du * A_v) / std::abs(v_A_du), tolerance);
  }

  // at u = 0
  {
    const double mu     = 0.5 * (material.mu1 * material.alpha1 + material.mu2 * material.alpha2);
    const double lambda = material.kappa - 2. / 3. * mu;
    VectorType   zero(u), residual(u);
    zero = 0.;
    op.evaluate_residual(residual, zero);
    reference_elasticity<dim, Number>(mapping,
                                      dof_handler,
                                      constraints,
                                      n_q_points_1d,
                                      lambda,
                                      mu,
                                      locally_owned_dofs,
                                      locally_relevant_dofs,
                                      MPI_COMM_WORLD,
                                      du,
                                      dst_ref,
                                      diag);
    check("vmult_linearized at u = 0 against linear elasticity", linearized, du);
  }
}



template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  test<dim, fe_degree, n_q_points_1d>("Mooney-Rivlin", {0.8, 2., -0.3, -2., 2.5});
  test<dim, fe_degree, n_q_points_1d>("Ogden", {0.9, 3.7, -0.2, -1.3, 2.5});
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>();
  test<2, 2>();
  test<3, 1>();
  test<3, 2>();
  // fe_degree + 2 points
  test<3, 2, 4>();
  // general kernel: too many points for the transformation to collocation
  test<3, 1, 3>();
}
