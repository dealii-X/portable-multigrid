#!/bin/bash
#SBATCH --job-name=hanging-node-kernels
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=00:10:00
#SBATCH --output=hanging-node-kernels-%j.out
#
# Run the deal.II tests hanging_node_kernels_01 and _02 (branch
# portable-hanging-nodes-line-kernel) on one GPU of JUPITER and compare their
# output with the expected output recorded on the CPU.
#
# Build on the login node against the deal.II that scripts/jupiter/env.sh
# points to, then submit, both from this directory:
#
#   source ../../scripts/jupiter/env.sh [DEAL_II_DIR]
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR
#   cmake --build build
#   sbatch run_sbatch.sh
#
# Inside an interactive allocation, `bash run_sbatch.sh` works as well.
# BUILD_DIR overrides the build directory.

set -u

BUILD_DIR=${BUILD_DIR:-build}

# module stack and LD_LIBRARY_PATH (DEAL_II_DIR from the environment or its
# default in env.sh)
source ../../scripts/jupiter/env.sh ${DEAL_II_DIR:-}

export KOKKOS_DEVICE_BACKEND=CUDA
export KOKKOS_DISABLE_OPENMP=true
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export CUDA_DEVICE_WAITS_ON_EXCEPTION=0

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

status=0
for test in hanging_node_kernels_01 hanging_node_kernels_02; do
  executable=$PWD/$BUILD_DIR/$test
  if [ ! -x "$executable" ]; then
    echo "no executable $executable, build first (see header of this script)" >&2
    exit 1
  fi

  # the tests write their log to ./output
  run_dir=$PWD/$BUILD_DIR/run/$test
  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  # --mpi=pspmix: plain srun starts singleton ranks on JUPITER
  (cd "$run_dir" && srun -n 1 --mpi=pspmix --gpus-per-task=1 "$executable")
  exit_code=$?

  echo "$test: exit code $exit_code"
  python3 compare_output.py "$run_dir/output" matrix_free_kokkos/$test.output
  compare_code=$?

  if [ $exit_code -eq 0 ] && [ $compare_code -eq 0 ]; then
    echo "$test: PASSED"
  else
    echo "$test: FAILED (output in $run_dir/output)"
    status=1
  fi
  echo
done

exit $status
