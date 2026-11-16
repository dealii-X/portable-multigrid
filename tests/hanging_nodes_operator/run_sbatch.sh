#!/bin/bash
#SBATCH --job-name=hanging-nodes-operator
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=results/operator-degree%a-%A.out
#SBATCH --array=1-6
#
# Laplace operator on deal.II's Portable::MatrixFree/FEEvaluation on an
# adaptively refined mesh, one job per degree, 2d and 3d, ball and random
# mesh, on one GPU of JUPITER. The first array task also writes the register,
# shared and stack usage of all kernels (the cell loops are
# Portable::internal::ApplyKernel<..., LocalLaplaceOperator<dim, degree>, ...>)
# to results/resource-usage-<label>-<job>.txt.
#
# Build once per deal.II to compare (master, and the branch whose
# FEEvaluation uses the batched hanging-node kernel), each in its own build
# directory, from this directory:
#
#   source ../../scripts/jupiter/env.sh [DEAL_II_DIR]
#   cmake -S . -B build-master -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR
#   cmake --build build-master
#   BUILD_DIR=build-master sbatch run_sbatch.sh
#
# and later the same with the other deal.II, e.g.
#
#   DEAL_II_DIR=<branch install> BUILD_DIR=build-batched sbatch run_sbatch.sh
#
# The label in the result files is the build directory. Slurm opens the
# output file before the script runs, so results/ must exist at submission;
# it is kept in git through results/.gitkeep. Override through the
# environment, e.g.
#
#   DIMS=3 MESHES=r N_DOFS=67108864 sbatch run_sbatch.sh
#
# Program arguments: dim degree n_dofs n_repetitions mesh.

set -u

DIMS=${DIMS:-"2 3"}
MESHES=${MESHES:-"b r"}
N_DOFS=${N_DOFS:-33554432}   # 2^25 dofs
N_REPETITIONS=${N_REPETITIONS:-50}
BUILD_DIR=${BUILD_DIR:-build-master}
degree=$SLURM_ARRAY_TASK_ID
label=$(basename "$BUILD_DIR")

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

echo "degree=$degree dims=$DIMS meshes=$MESHES n_dofs=$N_DOFS repetitions=$N_REPETITIONS"
echo "executable: $BUILD_DIR/program"
echo "deal.II:    $DEAL_II_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

if [ "$degree" = 1 ]; then
  resources=results/resource-usage-${label}-${SLURM_ARRAY_JOB_ID}.txt
  if command -v cu++filt > /dev/null; then
    cuobjdump --dump-resource-usage "$BUILD_DIR/program" | cu++filt > "$resources"
  else
    cuobjdump --dump-resource-usage "$BUILD_DIR/program" > "$resources"
  fi
  echo "resource usage written to $resources"
  echo
fi

status=0
for mesh in $MESHES; do
  for dim in $DIMS; do
    # --mpi=pspmix: plain srun starts singleton ranks on JUPITER
    srun -n 1 \
         --mpi=pspmix \
         --cpu-bind=cores \
         --gpu-bind=closest \
         --mem-bind=local \
         --gpus-per-task=1 \
         "$BUILD_DIR/program" "$dim" "$degree" "$N_DOFS" "$N_REPETITIONS" "$mesh" || status=1
  done
done

exit $status
