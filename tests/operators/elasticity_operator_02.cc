// Portable::ElasticityOperator on a deformed mesh, with communication/computation overlap.

#include "../tests.h"

#include "elasticity_operator_deformed.h"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_all(true);
}
