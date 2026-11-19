// CG with a geometric multigrid V-cycle of Portable::LaplaceOperatorBK3 level
// operators, GeometricTransfer and Chebyshev smoothing of degree 3: number of
// iterations to a relative residual of 1e-10.

#include "../tests.h"

#include "gmg_util.h"
#include "operators/portable_laplace_operator_bk3.h"

template <int dim, int fe_degree>
void
test(const unsigned int n_refinements, const unsigned int min_its, const unsigned int max_its)
{
  gmg_test::solve<dim, fe_degree, Portable::LaplaceOperatorBK3<dim, fe_degree, double>>(
    n_refinements, 3, min_its, max_its);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>(5, 7, 9);
  test<2, 2>(5, 7, 9);
  test<2, 4>(4, 7, 9);
  test<3, 1>(4, 6, 8);
  test<3, 2>(3, 7, 9);
  test<3, 3>(3, 7, 9);
}
