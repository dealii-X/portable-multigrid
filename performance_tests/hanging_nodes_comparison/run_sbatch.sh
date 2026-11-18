#!/bin/bash
#SBATCH --job-name=hanging-nodes-comparison
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=00:15:00
#SBATCH --output=results/hanging-nodes-degree%a-%A.out
#SBATCH --array=1-6
#
# Compare deal.II's cell-wise hanging-node kernel with the batched one of the
# deal.II branch batched-hanging-nodes (batch sizes 1, 2, 4, 8) on one GPU of
# JUPITER, one job per degree, 2d and 3d.
#
# Submit from this directory after building against the deal.II that
# scripts/jupiter/env.sh points to:
#
#   source ../../scripts/jupiter/env.sh [DEAL_II_DIR]
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR
#   cmake --build build
#   sbatch run_sbatch.sh
#
# Slurm opens the output file before the script runs, so results/ must exist
# at submission; it is kept in git through results/.gitkeep.
#
# The job array index is the FE_Q degree. The first array task also writes
# the register, shared and stack usage of all kernels to
# results/resource-usage-<job>.txt. Override through the environment, e.g.
#
#   DIMS=3 N_DOFS=268435456 sbatch run_sbatch.sh
#
# Program arguments: dim degree n_dofs n_repetitions.

set -u

DIMS=${DIMS:-"2 3"}
N_DOFS=${N_DOFS:-134217728}   # 2^27 dofs, 1 GiB per vector
N_REPETITIONS=${N_REPETITIONS:-20}
BUILD_DIR=${BUILD_DIR:-build}
degree=$SLURM_ARRAY_TASK_ID

# module stack, LD_LIBRARY_PATH and thread pinning (DEAL_II_DIR from the
# environment or its default in env.sh)
source ../../scripts/jupiter/env.sh ${DEAL_II_DIR:-}

export KOKKOS_DEVICE_BACKEND=CUDA
export KOKKOS_DISABLE_OPENMP=true
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export CUDA_DEVICE_WAITS_ON_EXCEPTION=0

if [ ! -x "$BUILD_DIR/program" ]; then
  echo "no executable $BUILD_DIR/program, build first (see header of this script)" >&2
  exit 1
fi

echo "degree=$degree dims=$DIMS n_dofs=$N_DOFS repetitions=$N_REPETITIONS"
echo "executable: $BUILD_DIR/program"
echo "deal.II:    $DEAL_II_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

if [ "$degree" = 1 ]; then
  resources=results/resource-usage-${SLURM_ARRAY_JOB_ID}.txt
  if command -v cu++filt > /dev/null; then
    cuobjdump --dump-resource-usage "$BUILD_DIR/program" | cu++filt > "$resources"
  else
    cuobjdump --dump-resource-usage "$BUILD_DIR/program" > "$resources"
  fi
  echo "resource usage written to $resources"
  echo
fi

status=0
for dim in $DIMS; do
  # --mpi=pspmix: plain srun starts singleton ranks on JUPITER
  srun -n 1 \
       --mpi=pspmix \
       --cpu-bind=cores \
       --gpu-bind=closest \
       --mem-bind=local \
       --gpus-per-task=1 \
       "$BUILD_DIR/program" "$dim" "$degree" "$N_DOFS" "$N_REPETITIONS" || status=1
done

exit $status
