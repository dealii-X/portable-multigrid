#!/bin/bash
# JUPITER environment for running the GPU binaries (source it, do not execute):
#
#   source scripts/jupiter/env.sh [DEAL_II_DIR]
#
# Loads the module stack the binaries were built with, puts the deal.II
# install and the Kokkos it was built against first on LD_LIBRARY_PATH
# (LD_LIBRARY_PATH beats RUNPATH) and sets the thread/pinning variables.
# DEAL_II_DIR defaults to $WORK/dealii/dealii-master.
#
# Correctness tests inside an allocation, from the repo root:
#
#   for t in laplace_bk3_deformed laplace_dg_deformed subdomain_laplace_deformed laplace_deformed vector_laplace_deformed; do
#     for np in 1 2; do
#       srun -n $np --cpu-bind=cores --gpu-bind=closest correctness_tests/$t/build-kokkos5/program > $t.np$np.out 2>&1
#       echo "$t np=$np exit=$? $(grep -E '^(PASSED|FAILED)' $t.np$np.out)"
#     done
#   done

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "source this file, do not execute it: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export DEAL_II_DIR=${1:-${DEAL_II_DIR:-$WORK/dealii/dealii-master}}

module --force purge
module load Stages/2026
module load GCCcore/.14.3.0
module load CMake/4.0.3
module load nvidia-compilers
module load OpenMPI # deal.II is built against OpenMPI; ParaStationMPI breaks the ABI
module load FlexiBLAS

# Kokkos library directory as recorded in deal.II's CMake config
kokkos_lib=$(grep -ho '/[^";]*/libkokkoscore[^";]*' "$DEAL_II_DIR"/lib*/cmake/deal.II/*.cmake 2>/dev/null | head -1 | xargs -r dirname)

export LD_LIBRARY_PATH=$DEAL_II_DIR/lib:$DEAL_II_DIR/lib64${kokkos_lib:+:$kokkos_lib}:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# exit-time abort (static Kokkos linked twice), skip the core files
ulimit -c 0

echo "DEAL_II_DIR=$DEAL_II_DIR"
echo "Kokkos lib: ${kokkos_lib:-not found in deal.II config, relying on \$WORK/sw/lib}"
