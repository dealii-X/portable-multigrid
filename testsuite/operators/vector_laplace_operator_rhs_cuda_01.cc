// Portable::VectorLaplaceOperator::compute_rhs_cuda() against
// compute_rhs_bk4() (checked against FEValues in
// vector_laplace_operator_rhs_01) for a unit load. 3D only; needs a CUDA
// build (nvcc).

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include "../tests.h"

#include "cube_random_util.h"
#include "operators/portable_vector_laplace_operator.h"

using namespace dealii;
using namespace cube_random;

template <int fe_degree, int n_components, int n_q_points_1d = fe_degree + 1>
void
test()
{
  constexpr int dim = 3;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << ", n_components = " << n_components
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  make_cube(triangulation);

  const FESystem<dim> fe(FE_Q<dim>{fe_degree}, n_components);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const AffineConstraints<double> constraints = make_dirichlet_constraints(dof_handler);

  const Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double, n_q_points_1d> op(
    dof_handler, constraints, false);

  VectorType rhs, rhs_ref;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(rhs_ref);

  op.compute_rhs_bk4(rhs_ref);
  op.compute_rhs_cuda(rhs);
  Kokkos::fence();

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
  report("compute_rhs_cuda",
         relative_difference(rhs, rhs_ref, dof_handler.locally_owned_dofs(), constraints),
         1e-10);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 3>();
  test<3, 2>();
  test<2, 3, 4>();
  test<3, 2, 5>();
}
