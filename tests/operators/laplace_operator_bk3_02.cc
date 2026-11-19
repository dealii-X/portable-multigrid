// Portable::LaplaceOperatorBK3 with the G tensors computed on the device:
// flat manifold, vertices moved by a smooth map after the refinement, so
// every cell is a general (bi/tri)linear cell. vmult(), compute_diagonal()
// and compute_rhs() against the CPU FEValues reference (MappingQ through the
// support points).
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_laplace_operator_bk3.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
void
test()
{
  using VectorType = DeviceVectorType<Number>;
  using Operator   = Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d>;

  const double tolerance = std::is_same_v<Number, double> ? 1e-12 : 1e-5;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << ", "
          << (std::is_same_v<Number, double> ? "double" : "float") << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(dim == 2 ? 3 : 2);
  GridTools::transform(&CurvedShearedManifold<dim>::map_forward, triangulation);

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

  Operator op(dof_handler, constraints, false);

  deallog << "n_dofs = " << dof_handler.n_dofs()
          << ", geometry on device = " << (op.geometry_on_device() ? "true" : "false")
          << std::endl;

  VectorType src, dst, dst_ref, diag_ref, rhs, rhs_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_ref);
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(rhs_ref);

  fill_source(src, locally_owned_dofs);

  reference_laplace<dim, Number>(mapping,
                                 dof_handler,
                                 constraints,
                                 n_q_points_1d,
                                 locally_owned_dofs,
                                 locally_relevant_dofs,
                                 MPI_COMM_WORLD,
                                 src,
                                 dst_ref,
                                 diag_ref);

  dst = 0.;
  op.vmult(dst, src);
  Kokkos::fence();
  report("vmult", relative_l2_difference(dst, dst_ref), tolerance);

  op.compute_diagonal();
  Kokkos::fence();
  report("compute_diagonal",
         max_relative_diagonal_error(op.get_matrix_diagonal_inverse()->get_vector(),
                                     diag_ref,
                                     locally_owned_dofs,
                                     MPI_COMM_WORLD),
         tolerance);

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
  report("compute_rhs", relative_l2_difference(rhs, rhs_ref), tolerance);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1, double>();
  test<2, 2, double>();
  test<2, 3, double>();
  test<2, 4, double>();
  test<2, 2, double, 4>();
  test<3, 1, double>();
  test<3, 2, double>();
  test<3, 3, double>();
  test<3, 4, double>();
  test<3, 2, double, 4>();
  test<3, 3, double, 5>();
  test<3, 2, float>();
}
