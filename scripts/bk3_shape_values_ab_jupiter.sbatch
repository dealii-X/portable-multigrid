#!/bin/bash
#SBATCH --job-name=bk3-shape-ab
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=bk3-shape-ab-%j.out

# poisson_cube_bk3 (performance_tests/poisson_cube_multigrid) with the cell kernel of
# LaplaceOperatorBK3::vmult() chosen by BK3_VMULT_KERNEL:
#   plain           hand-written kernel, shape values in the team scratch
#   plain_constant  hand-written kernel, shape values in the kernel closure
#   abstracted      the default kernel, for reference only (other launch configuration)
# One binary on one node; the kernels alternate within each repetition.
#   DEGREES="1 2 3 4" KERNELS="plain plain_constant abstracted" REPS=2 GPUS=1 sbatch <this script>
# Doubling mesh, 5 + 5 smoothing, float V-cycle, MIN_PER_GPU .. MAX_PER_GPU DoFs, 1 rank
# per GPU. One JSON per run in OUT; at the end plain_constant against plain
# (compare_baseline.py: ratio < 1 means plain_constant is faster, cg_its must agree).

# checkout and binary, override with REPO=... BIN=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-portable-multigrid}
BIN=${BIN:-$REPO/performance_tests/poisson_cube_multigrid/build/poisson_cube_bk3}

DEGREES=${DEGREES:-"1 2 3 4"}
KERNELS=${KERNELS:-"plain plain_constant abstracted"}
REPS=${REPS:-2}
GPUS=${GPUS:-1}
MIN_PER_GPU=${MIN_PER_GPU:-100000}
MAX_PER_GPU=${MAX_PER_GPU:-40000000}
MIN=$((MIN_PER_GPU * GPUS))
MAX=$((MAX_PER_GPU * GPUS))
OUT=${OUT:-$PWD/bk3-shape-ab-${SLURM_JOB_ID:-local}}

source $REPO/scripts/jupiter/env.sh
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

mkdir -p $OUT
ldd $BIN | grep -E 'libdeal_II|libkokkoscore'
git -C $REPO log -1 --format='commit %h %s'
echo "GPUs: $GPUS, sizes $MIN .. $MAX DoFs, degrees: $DEGREES, kernels: $KERNELS, repetitions: $REPS"
echo "JSON files: $OUT"

opts="--mpi=pspmix --ntasks=$GPUS --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"

echo "placement:"
srun $opts bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V

for degree in $DEGREES; do
  for rep in $(seq 1 $REPS); do
    for kernel in $KERNELS; do
      echo
      echo "=============== kernel=$kernel degree=$degree repetition=$rep"
      BK3_VMULT_KERNEL=$kernel timeout 1200 srun $opts $BIN $degree $MIN $MAX 5 5 d 0 \
        --json $OUT/${kernel}_p${degree}_r${rep}.json
    done
  done
done

for degree in $DEGREES; do
  for rep in $(seq 1 $REPS); do
    new=$OUT/plain_constant_p${degree}_r${rep}.json
    old=$OUT/plain_p${degree}_r${rep}.json
    if [ -f $new ] && [ -f $old ]; then
      echo
      echo "=============== plain_constant / plain, degree=$degree repetition=$rep"
      python3 $REPO/performance_tests/compare_baseline.py $new $old --tolerance 0.05
    fi
  done
done
