// Portable::VectorLaplaceOperator::vmult_cuda(), the plain fp64 GEMM-in-registers kernel of
// bk4_cuda_plain_kernels.cuh, against vmult_dealii() on a random vector. 3D only; needs a CUDA
// build (nvcc).

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include "../tests.h"

#include "cube_random_util.h"
#include "operators/portable_vector_laplace_operator.h"

using namespace dealii;
using namespace cube_random;

template <int fe_degree, int n_components>
void
test()
{
  constexpr int dim = 3;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << ", n_components = " << n_components
          << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  make_cube(triangulation);

  const FESystem<dim> fe(FE_Q<dim>{fe_degree}, n_components);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const AffineConstraints<double> constraints = make_dirichlet_constraints(dof_handler);

  const Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double> op(dof_handler,
                                                                                 constraints,
                                                                                 false);

  VectorType src, dst, dst_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  fill_random(src, dof_handler.locally_owned_dofs(), constraints);

  op.vmult_dealii(dst_ref, src);
  op.vmult_cuda(dst, src);
  Kokkos::fence();

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
  report("vmult_cuda",
         relative_difference(dst, dst_ref, dof_handler.locally_owned_dofs(), constraints),
         1e-10);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 3>();
  test<3, 2>();
  test<4, 2>();
}
