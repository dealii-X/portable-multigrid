// Portable::LaplaceOperator::vmult_batched(), vmult_fused() and vmult_fused_G()
// (the batched FEEvaluation copied from deal.II, without and with
// evaluate_and_apply(), and with the BK3 geometry G) on a
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
