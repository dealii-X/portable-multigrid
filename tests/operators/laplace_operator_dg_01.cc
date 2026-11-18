// Portable::LaplaceOperatorDG on a deformed mesh, without communication/computation overlap.

#include "../tests.h"

#include "laplace_operator_dg_deformed.h"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_all(false, true);
}
