// Portable::ElasticityOperator::vmult_benchmark(), the kernel of the dealii-X
// benchmarks with the geometry of Portable::MatrixFree and gather/scatter on
// the global vectors, on the curved, sheared mesh of deformed_mesh_util.h
// against the CPU FEValues reference, in 3d with fe_degree + 1 and
// fe_degree + 2 (the benchmark's) quadrature points per direction.

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

template <int fe_degree, int n_q_points_1d>
void
test()
{
  constexpr int dim = 3;
  using Number      = double;
  using VectorType  = DeviceVectorType<Number>;
  using Material    = Portable::LinearElasticMaterial<dim, Number>;
  using Operator    = Portable::ElasticityOperator<dim, fe_degree, Number, Material, n_q_points_1d>;

  const Material material{1.7, 0.6};

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, 2);

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

  const Operator op(dof_handler, constraints, material, false);

  VectorType src, dst, dst_ref, diag_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_ref);

  fill_source(src, locally_owned_dofs, &constraints);

  reference_elasticity<dim, Number>(mapping,
                                    dof_handler,
                                    constraints,
                                    n_q_points_1d,
                                    material.lambda,
                                    material.mu,
                                    locally_owned_dofs,
                                    locally_relevant_dofs,
                                    MPI_COMM_WORLD,
                                    src,
                                    dst_ref,
                                    diag_ref);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  op.vmult_benchmark(dst, src);
  Kokkos::fence();
  report("vmult_benchmark", relative_l2_difference(dst, dst_ref), 1e-12);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<1, 2>();
  test<2, 3>();
  test<3, 4>();
  // the benchmark's fe_degree + 2 points
  test<1, 3>();
  test<2, 4>();
}
