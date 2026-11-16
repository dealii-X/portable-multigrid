#!/bin/bash
#SBATCH --job-name=mg-transfer-comparison
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=00:30:00
#SBATCH --output=results/transfer-degree%a-%A.out
#SBATCH --array=1-6
#
# Compare the three transfer kernels (installed deal.II, the batched copy to
# be PR'd, this repo's BK1) on one GPU of JUPITER, one job per degree.
#
# Submit from this directory after building against the deal.II that
# scripts/jupiter/env.sh points to:
#
#   source ../../scripts/jupiter/env.sh [DEAL_II_DIR]
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR
#   cmake --build build
#   mkdir -p results && sbatch run_sbatch.sh
#
# Slurm opens the output file before the script runs, so results/ must exist
# at submission; it is kept in git through results/.gitkeep.
#
# The job array index is the FE_Q degree (1 to 6, what deal.II's Portable
# transfer dispatches). Override the dimension, mesh sizes and build directory
# through the environment, e.g.
#
#   DIM=2 MAX_SIZE=100000000 sbatch run_sbatch.sh
#   NTASKS=4 sbatch --ntasks-per-node=4 --gres=gpu:4 run_sbatch.sh
#
# Program arguments: dimension degree minsize maxsize doubling.

set -u

DIM=${DIM:-3}
MIN_SIZE=${MIN_SIZE:-1000}
# Default size cap from the OOM kills seen on a GH200 node with one rank:
# 2^26 fine cells (the two triangulations on one rank) at degrees 1 and 2,
# and 1e9 fine dofs (per-cell indices and weights of four transfers) at
# degree 5. Allow 2^25 fine cells, i.e. 2^25 * degree^3 dofs, at most 6e8.
degree=$SLURM_ARRAY_TASK_ID
default_max_size=$((33554432 * degree * degree * degree))
[ "$default_max_size" -gt 600000000 ] && default_max_size=600000000
MAX_SIZE=${MAX_SIZE:-$default_max_size}
MESH=${MESH:-d}          # d: doubling mesh, s: square mesh
BUILD_DIR=${BUILD_DIR:-build}
NTASKS=${NTASKS:-$SLURM_NTASKS_PER_NODE}

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

echo "degree=$degree dim=$DIM sizes=[$MIN_SIZE,$MAX_SIZE] mesh=$MESH ntasks=$NTASKS"
echo "executable: $BUILD_DIR/program"
echo "deal.II:    $DEAL_II_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# --mpi=pspmix: plain srun starts singleton ranks on JUPITER
srun -n "$NTASKS" \
     --mpi=pspmix \
     --cpu-bind=cores \
     --gpu-bind=closest \
     --mem-bind=local \
     --gpus-per-task=1 \
     "$BUILD_DIR/program" "$DIM" "$degree" "$MIN_SIZE" "$MAX_SIZE" "$MESH"
