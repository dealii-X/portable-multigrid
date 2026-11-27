// Portable::ElasticityOperator with the Saint Venant-Kirchhoff material: the
// diagonal of the operator linearized at u by compute_linearized_diagonal(),
// against the CPU FEValues reference of the linearized operator applied to
// the unit vectors, on the curved, sheared mesh of deformed_mesh_util.h.

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
test(const unsigned int n_refinements)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Material   = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using Operator   = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  const Material material{1.7, 0.6};

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, n_refinements);

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

  VectorType u, residual, unit, column, diagonal, diagonal_ref;
  for (VectorType *vec : {&u, &residual, &unit, &column, &diagonal, &diagonal_ref})
    op.initialize_dof_vector(*vec);

  fill_source(u, locally_owned_dofs, &constraints);
  u *= 0.05;

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  op.evaluate_residual(residual, u);
  op.compute_linearized_diagonal(diagonal);

  // reference: entry i of A e_i
  LinearAlgebra::ReadWriteVector<Number> rw_diagonal(locally_owned_dofs);
  for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
    {
      LinearAlgebra::ReadWriteVector<Number> rw_unit(locally_owned_dofs);
      for (const auto j : locally_owned_dofs)
        rw_unit(j) = (j == i) ? 1. : 0.;
      unit.import_elements(rw_unit, VectorOperation::insert);

      reference_svk_linearized<dim, Number>(mapping,
                                            dof_handler,
                                            constraints,
                                            n_q_points_1d,
                                            material.lambda,
                                            material.mu,
                                            locally_owned_dofs,
                                            locally_relevant_dofs,
                                            MPI_COMM_WORLD,
                                            u,
                                            unit,
                                            column);

      // collective
      LinearAlgebra::ReadWriteVector<Number> rw_column(locally_owned_dofs);
      rw_column.import_elements(column, VectorOperation::insert);
      if (locally_owned_dofs.is_element(i))
        rw_diagonal(i) = rw_column(i);
    }
  diagonal_ref.import_elements(rw_diagonal, VectorOperation::insert);

  report("compute_linearized_diagonal", relative_l2_difference(diagonal, diagonal_ref), 1e-12);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>(2);
  test<2, 2>(2);
  test<3, 1>(1);
  test<3, 2>(1);
  // fe_degree + 2 points
  test<3, 2, 4>(1);
}
