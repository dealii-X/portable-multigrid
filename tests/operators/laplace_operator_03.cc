// Portable::LaplaceOperator::vmult_batched() and vmult_fused() (the batched
// FEEvaluation copied from deal.II, without and with evaluate_and_apply()) on a
// deformed mesh, without communication/computation overlap.

#include "../tests.h"

#include "laplace_operator_deformed.h"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_all<true>(false);
}
