// Portable::ElasticityOperator with the Saint Venant-Kirchhoff material: the
// residual R(u) by evaluate_residual() and evaluate_residual_fused(), which
// store grad u, and the operator linearized at u by vmult_linearized() and
// vmult_linearized_fused() on the curved, sheared mesh of deformed_mesh_util.h
// against the CPU FEValues references. Further checks of the linearization:
// - R is cubic in u, so the error of the central difference
//   (R(u + eps du) - R(u - eps du)) / (2 eps) is exactly quadratic in eps:
//   halving eps divides it by 4
// - the linearized operator is symmetric
// - at u = 0 it is linear elasticity

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

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  constexpr double tolerance = 1e-12;

  const Material material{1.7, 0.6};

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

  VectorType u, du, v, dst, dst_ref, diag;
  for (VectorType *vec : {&u, &du, &v, &dst, &dst_ref, &diag})
    op.initialize_dof_vector(*vec);

  // v as u, du another pattern, all zero on the constrained DoFs
  fill_source(u, locally_owned_dofs, &constraints);
  v = u;
  u *= 0.05;
  {
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = constraints.is_constrained(i) ? 0. : std::cos(0.29 * static_cast<double>(i + 2));
    du.import_elements(rw, VectorOperation::insert);
  }

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  const auto reference_linearized = [&](const VectorType &point, const VectorType &direction)
  {
    reference_svk_linearized<dim, Number>(mapping,
                                          dof_handler,
                                          constraints,
                                          n_q_points_1d,
                                          material.lambda,
                                          material.mu,
                                          locally_owned_dofs,
                                          locally_relevant_dofs,
                                          MPI_COMM_WORLD,
                                          point,
                                          direction,
                                          dst_ref);
  };

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

  // residual and linearization at u, with grad u stored by either variant
  reference_svk_residual<dim, Number>(mapping,
                                      dof_handler,
                                      constraints,
                                      n_q_points_1d,
                                      material.lambda,
                                      material.mu,
                                      locally_owned_dofs,
                                      locally_relevant_dofs,
                                      MPI_COMM_WORLD,
                                      u,
                                      dst_ref);
  check(
    "evaluate_residual_fused",
    [&](VectorType &d, const VectorType &s) { op.evaluate_residual_fused(d, s); },
    u);
  check(
    "evaluate_residual",
    [&](VectorType &d, const VectorType &s) { op.evaluate_residual(d, s); },
    u);

  reference_linearized(u, du);
  check("vmult_linearized", linearized, du);
  check("vmult_linearized_fused", linearized_fused, du);
  {
    VectorType residual(u);
    op.evaluate_residual_fused(residual, u);
  }
  check("vmult_linearized after evaluate_residual_fused", linearized, du);
  check("vmult_linearized_fused after evaluate_residual_fused", linearized_fused, du);

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

    const double ratio = difference_error(1e-2) / difference_error(5e-3);
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
    VectorType zero(u), residual(u);
    zero = 0.;
    op.evaluate_residual(residual, zero);
    reference_elasticity<dim, Number>(mapping,
                                      dof_handler,
                                      constraints,
                                      n_q_points_1d,
                                      material.lambda,
                                      material.mu,
                                      locally_owned_dofs,
                                      locally_relevant_dofs,
                                      MPI_COMM_WORLD,
                                      du,
                                      dst_ref,
                                      diag);
    check("vmult_linearized at u = 0 against linear elasticity", linearized, du);
  }
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
