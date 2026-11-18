// Portable::LaplaceOperatorBK3::vmult_tensor_core() (the n_components = 1
// case of the FP64 tensor-core kernel of bk4_cuda_kernels.cuh) against
// vmult() on a random vector, for p + 1 and p + 2 quadrature points. 3D and
// double only; needs a CUDA build (nvcc) on sm_80 or newer.

#include <deal.II/fe/fe_q.h>

#include "../tests.h"

#include "cube_random_util.h"
#include "operators/portable_laplace_operator_bk3.h"

using namespace dealii;
using namespace cube_random;

template <int fe_degree, int n_q_points_1d = fe_degree + 1>
void
test()
{
  constexpr int dim = 3;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  make_cube(triangulation);

  const FE_Q<dim> fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const AffineConstraints<double> constraints = make_dirichlet_constraints(dof_handler);

  const Portable::LaplaceOperatorBK3<dim, fe_degree, double, n_q_points_1d> op(dof_handler,
                                                                               constraints,
                                                                               false);

  VectorType src, dst, dst_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  fill_random(src, dof_handler.locally_owned_dofs(), constraints);

  op.vmult(dst_ref, src);
  op.vmult_tensor_core(dst, src);
  Kokkos::fence();

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
  report("vmult_tensor_core",
         relative_difference(dst, dst_ref, dof_handler.locally_owned_dofs(), constraints),
         1e-10);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2>();
  test<3>();
  test<2, 4>();
  test<3, 5>();
}
