#!/bin/bash
#SBATCH --job-name=mg-refs
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=mg-refs-%j.out

# GPU-only and CPU-only references for the hybrid program on the nodes of the job:
#   1. poisson_cube_bk3 of this branch, 1 rank per GPU (4 per node)
#   2. poisson_cube_cpu, 1 rank per core (288 per node), block placement
# Q4, doubling mesh, 5 + 5 smoothing, float V-cycle, the cycles of the hybrid
# doubling job (run_sbatch_jupiter_doubling.sh) for the same number of nodes.
# Several nodes: sbatch --nodes=<N> <this script>. MIN_PER_NODE / MAX_PER_NODE (DoFs)
# change the size range as in the doubling job.

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
CPU_BIN=$REPO/poisson_cube_cpu/$BUILD/program
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program

NODES=${SLURM_JOB_NUM_NODES:-1}

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $CPU_BIN | grep -E 'libdeal_II|libkokkoscore'

MIN_PER_NODE=${MIN_PER_NODE:-200000}
MAX_PER_NODE=${MAX_PER_NODE:-70000000}
ARGS="4 $((MIN_PER_NODE * NODES)) $((MAX_PER_NODE * NODES)) 5 5 d 0"

# run <label> <srun options> <program and arguments>; the first and last 4 ranks
# show the placement
run() {
  local label=$1 opts=$2; shift 2
  echo
  echo "=============== $label: srun $opts $*"
  echo "placement, first and last ranks:"
  srun $opts bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V | sed -n '1,4p;$p'
  timeout 2400 srun $opts "$@"
}

echo "######## 1. GPU only, bk3, $((4 * NODES)) ranks on $NODES node(s)"
run gpu "--mpi=pspmix --nodes=$NODES --ntasks=$((4 * NODES)) --ntasks-per-node=4 --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local" $BK3_BIN $ARGS

echo "######## 2. CPU only, $((288 * NODES)) ranks on $NODES node(s), block placement"
run cpu "--mpi=pspmix --nodes=$NODES --ntasks=$((288 * NODES)) --ntasks-per-node=288 --cpus-per-task=1 --distribution=block:block --cpu-bind=cores --mem-bind=local" $CPU_BIN $ARGS
