// Portable::LaplaceOperatorBK3::compute_rhs() for a unit load on a curved,
// sheared mesh against a CPU FEValues assembly with the same MappingQ, for
// p + 1 and p + 2 quadrature points.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_laplace_operator_bk3.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  using Number = double;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FE_Q<dim>     fe(fe_degree);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(), constraints);
  constraints.close();

  const Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d> op(dof_handler,
                                                                               constraints,
                                                                               false);

  DeviceVectorType<Number> rhs, rhs_ref;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(rhs_ref);

  op.compute_rhs(rhs);
  Kokkos::fence();

  reference_unit_rhs<dim, Number>(mapping,
                                  dof_handler,
                                  constraints,
                                  n_q_points_1d,
                                  locally_owned_dofs,
                                  locally_relevant_dofs,
                                  MPI_COMM_WORLD,
                                  rhs_ref);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
  report("compute_rhs", relative_l2_difference(rhs, rhs_ref), 1e-12);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>();
  test<2, 2>();
  test<2, 3>();
  test<3, 1>();
  test<3, 2>();
  test<3, 3>();
  test<3, 2, 4>();
  test<3, 3, 5>();
}
