// CG with a polynomial multigrid V-cycle (degrees 1, ..., p on one mesh) of
// Portable::LaplaceOperatorBK3 level operators, PolynomialTransfer and Chebyshev
// smoothing of degree 3: number of iterations to a relative residual of
// 1e-10.

#include "../tests.h"

#include "operators/portable_laplace_operator_bk3.h"
#include "pmg_util.h"

template <int dim, int degree>
struct Operator
{
  using type = Portable::LaplaceOperatorBK3<dim, degree, double>;
};

template <int dim, int fe_degree>
void
test(const unsigned int n_refinements, const unsigned int min_its, const unsigned int max_its)
{
  pmg_test::solve<dim, fe_degree, Operator>(n_refinements, 3, min_its, max_its);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 2>(4, 7, 9);
  test<2, 4>(3, 7, 9);
  test<3, 2>(3, 7, 9);
  test<3, 3>(2, 7, 9);
  test<3, 4>(2, 7, 9);
}
