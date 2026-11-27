// Portable::ElasticityOperator with the Ogden material: the residual R(u) by
// evaluate_residual() and evaluate_residual_fused() on the curved, sheared
// mesh of deformed_mesh_util.h against the CPU FEValues reference. With
// (alpha1, alpha2) = (2, -2) Ogden is Mooney-Rivlin, whose stress the
// reference evaluates in closed form from the invariants of C (2D: plane
// strain, C_33 = 1):
//   S = mu1 J^(-2/3) (I - I1 / 3 C^-1)
//     - mu2 J^(-4/3) (I1 I - C - 2 / 3 I2 C^-1) + kappa / 2 (J^2 - 1) C^-1

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
mooney_rivlin(const double mu1, const double mu2, const double kappa, const Tensor<2, dim> &grad_u)
{
  Tensor<2, 3> F;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      F[i][j] = grad_u[i][j];
  for (unsigned int d = 0; d < 3; ++d)
    F[d][d] += 1.;
  const Tensor<2, 3> I     = unit_symmetric_tensor<3>();
  const Tensor<2, 3> C     = transpose(F) * F;
  const Tensor<2, 3> C_inv = invert(C);
  const double       J     = determinant(F);
  const double       I1    = trace(C);
  const double       I2    = 0.5 * (I1 * I1 - trace(C * C));

  const Tensor<2, 3> S = mu1 * std::pow(J, -2. / 3.) * (I - I1 / 3. * C_inv) -
                         mu2 * std::pow(J, -4. / 3.) * (I1 * I - C - 2. / 3. * I2 * C_inv) +
                         0.5 * kappa * (J * J - 1.) * C_inv;
  const Tensor<2, 3> P = F * S;

  Tensor<2, dim> result;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      result[i][j] = P[i][j];
  return result;
}

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::OgdenMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  constexpr double tolerance = 1e-12;

  const Material material{0.8, 2., -0.3, -2., 2.5};

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
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

  VectorType u, dst, dst_ref;
  for (VectorType *vec : {&u, &dst, &dst_ref})
    op.initialize_dof_vector(*vec);

  fill_source(u, locally_owned_dofs, &constraints);
  // smaller than in elasticity_operator_06: J > 0 at all points
  u *= 0.01;

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  const auto check = [&](const std::string &name, auto &&apply)
  {
    dst = 0.;
    apply(dst, u);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);
  };

  reference_hyperelastic_residual<dim, Number>(
    mapping,
    dof_handler,
    constraints,
    n_q_points_1d,
    [&](const Tensor<2, dim> &grad_u)
    { return mooney_rivlin<dim>(material.mu1, material.mu2, material.kappa, grad_u); },
    locally_owned_dofs,
    locally_relevant_dofs,
    MPI_COMM_WORLD,
    u,
    dst_ref);
  check("evaluate_residual_fused",
        [&](VectorType &d, const VectorType &s) { op.evaluate_residual_fused(d, s); });
  check("evaluate_residual",
        [&](VectorType &d, const VectorType &s) { op.evaluate_residual(d, s); });
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
