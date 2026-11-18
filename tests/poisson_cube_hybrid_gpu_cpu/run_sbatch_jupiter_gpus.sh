#!/bin/bash
#SBATCH --job-name=mg-gpus
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=mg-gpus-%j.out

# The three variants on 1 or 2 of the 4 GPUs of a node (and the Grace cores next to
# them), to complete the series 1, 2, 4 GPUs (one node), 8, 16 GPUs (two, four nodes):
# what does the communication between GPUs of the same node cost?
#   GPUS=1 sbatch --job-name=mg-1gpu --output=mg-1gpu-%j.out <this script>
#   GPUS=2 sbatch --job-name=mg-2gpu --output=mg-2gpu-%j.out <this script>
# Q4, doubling mesh, 5 + 5 smoothing, float V-cycle, the same sizes per GPU as on the
# full nodes (MIN_PER_GPU .. MAX_PER_GPU DoFs, default 70k .. 135M per GPU).
#   gpu     poisson_cube_bk3, 1 rank per GPU
#   cpu     poisson_cube_cpu, 72 ranks per GPU (the cores of its Grace), block placement
#   hybrid  K ranks per GPU, switch_dofs = 8000: K = 18 on all sizes, K = 4 on the
#           sizes from 30M DoFs per GPU
# PARTS="gpu cpu hybrid" selects the parts, e.g. PARTS=gpu for the GPU-only runs alone.
# The ranks fill the first GPUS Graces (block placement); each run prints its placement.

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
CPU_BIN=$REPO/poisson_cube_cpu/$BUILD/program
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program
HYB_BIN=$REPO/poisson_cube_hybrid_gpu_cpu/$BUILD/program

GPUS=${GPUS:-1}
PARTS=${PARTS:-"gpu cpu hybrid"}
MIN_PER_GPU=${MIN_PER_GPU:-60000}
MAX_PER_GPU=${MAX_PER_GPU:-140000000}
MIN=$((MIN_PER_GPU * GPUS))
MAX=$((MAX_PER_GPU * GPUS))

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $HYB_BIN | grep -E 'libdeal_II|libkokkoscore'
echo "GPUs used: $GPUS, sizes $MIN .. $MAX DoFs, parts: $PARTS"

# run <label> <ranks per GPU> <program and arguments>: the first and last ranks show
# the placement
run() {
  local label=$1 k=$2; shift 2
  local opts="--mpi=pspmix --ntasks=$((k * GPUS)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== $label: srun $opts $*"
  echo "placement, first and last ranks:"
  srun $opts bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V | sed -n '1,2p;$p'
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 2400 srun $opts "$@"
  echo "srun elapsed: $((SECONDS - t0)) s"
}

for part in $PARTS; do
  case $part in
    gpu)
      echo "######## GPU only, bk3, $GPUS rank(s)"
      run gpu 1 $BK3_BIN 4 $MIN $MAX 5 5 d 0
      ;;
    cpu)
      echo "######## CPU only, $((72 * GPUS)) ranks, block placement"
      run cpu 72 $CPU_BIN 4 $MIN $MAX 5 5 d 0
      ;;
    hybrid)
      # see run_sbatch_jupiter_doubling.sh; the program warns that SLURM_GPUS_ON_NODE
      # (4) differs from the number of GPU ranks, which is intended here
      export UCX_NUM_PPN=4
      echo "######## hybrid, 18 ranks per GPU, switch_dofs 8000"
      run hybrid-k18 18 $HYB_BIN 18 4 $MIN $MAX 5 5 d 8000 0
      echo "######## hybrid, 4 ranks per GPU, switch_dofs 8000, sizes from 30M DoFs per GPU"
      run hybrid-k4 4 $HYB_BIN 4 4 $((30000000 * GPUS)) $MAX 5 5 d 8000 0
      unset UCX_NUM_PPN
      ;;
    *)
      echo "unknown part $part"
      ;;
  esac
done
