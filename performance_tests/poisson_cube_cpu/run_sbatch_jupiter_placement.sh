#!/bin/bash
#SBATCH --job-name=cpu-placement
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=cpu-placement-%j.out

# Why are the CPU-only cycles of the hybrid program on 288 ranks 4 - 5 ms faster than
# poisson_cube_cpu on 288 ranks (job 2843214)? Same computation time per level, but
# the ghost exchange was 1.3 - 4x slower on every level in job 2843214.
#   1. poisson_cube_cpu, launched as in job 2843214 (no --distribution)
#   2. the same with --distribution=block:block (as the hybrid program is launched)
#   3. as 2 with UCX_NUM_PPN=4
#   4. poisson_cube_bk3 of this branch on 4 GPUs: the GPU-only reference for the
#      hybrid program (job 2843214 used the bk3 of the gpu-vs-cpu branch)
# All: Q4, doubling mesh, cycles 12..20, 5 + 5 smoothing, float V-cycle.

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
CPU_BIN=$REPO/poisson_cube_cpu/$BUILD/program
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $CPU_BIN | grep -E 'libdeal_II|libkokkoscore'

ARGS="4 200000 70000000 5 5 d 0"
CPU="--mpi=pspmix --ntasks=288 --cpus-per-task=1 --cpu-bind=cores --mem-bind=local"

# run <label> <srun options> -- <program and arguments>; the first 8 ranks show the placement
run() {
  local label=$1 opts=$2; shift 2
  echo
  echo "=============== $label: srun $opts $*"
  echo "placement of ranks 0..7:"
  srun $opts bash -c 'echo "rank $SLURM_PROCID gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V | head -8
  timeout 900 srun $opts "$@"
}

echo "######## 1. CPU, 288 ranks, launched as in job 2843214"
run cpu-default "$CPU" $CPU_BIN $ARGS

echo "######## 2. CPU, 288 ranks, block placement"
run cpu-block "$CPU --distribution=block:block" $CPU_BIN $ARGS

echo "######## 3. CPU, 288 ranks, block placement, UCX_NUM_PPN=4"
UCX_NUM_PPN=4 run cpu-block-ppn4 "$CPU --distribution=block:block" $CPU_BIN $ARGS

echo "######## 4. GPU only, bk3 of this branch, 1 rank per GPU"
run gpu "--mpi=pspmix --ntasks=4 --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local" $BK3_BIN $ARGS
