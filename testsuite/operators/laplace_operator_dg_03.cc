// Portable::LaplaceOperatorDG on the axis-aligned unit cube (check of the reference itself).

#include "../tests.h"

#include "laplace_operator_dg_deformed.h"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_all(false, false);
}
