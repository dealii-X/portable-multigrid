#!/bin/bash
#SBATCH --job-name=dealii-ab
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --output=dealii-ab-%j.out

# Old against new deal.II on the same nodes: dealii-neon-b35ee29762 (all results up to
# 2026-11-13) and dealii-neon d3f9f72c69 (with PR 20339, one Kokkos::fence per
# compress() with device-aware MPI). Each variant runs its own binary, built against
# its own library:
#   old: program-b35ee29762 (copied before the rebuild), new: program
# Per variant: bk3 GPU-only (1 rank per GPU) and the hybrid with K ranks per GPU,
# sleeping wait 50 us in both (the old hybrid binary defaults to the blocking wait).
# Order old new new old, so that a drift of the nodes cancels in the mean.
# Q4, doubling mesh, 5 + 5 smoothing, float V-cycle.
# Several nodes: sbatch --nodes=<N> --time=... <this script>; the size range grows
# with N (weak scaling) as in run_sbatch_jupiter_doubling.sh.

REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
OLD_TAG=${OLD_TAG:-b35ee29762}
LIB_OLD=${LIB_OLD:-/e/project1/dealii-x/ivan/sw/dealii/dealii-neon-$OLD_TAG/lib}
LIB_NEW=${LIB_NEW:-/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib}
K=${K:-18}
SWITCH_DOFS=${SWITCH_DOFS:-8000}
ORDER=${ORDER:-"old new new old"}
MIN_PER_NODE=${MIN_PER_NODE:-200000}
MAX_PER_NODE=${MAX_PER_NODE:-70000000}
NODES=${SLURM_JOB_NUM_NODES:-1}

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

BASE_LD=/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
# see run_sbatch_jupiter_doubling.sh
export UCX_NUM_PPN=4
export HYBRID_IDLE_SLEEP_US=50

OUT=$PWD/dealii-ab-$SLURM_JOB_ID
mkdir -p $OUT
MIN=$((MIN_PER_NODE * NODES))
MAX=$((MAX_PER_NODE * NODES))

# binary <program dir> <variant>
binary() {
  if [ "$2" = old ]; then echo $REPO/$1/$BUILD/program-$OLD_TAG; else echo $REPO/$1/$BUILD/program; fi
}

for v in old new; do
  lib=$LIB_OLD; [ $v = new ] && lib=$LIB_NEW
  for p in poisson_cube_bk3 poisson_cube_hybrid_gpu_cpu; do
    b=$(binary $p $v)
    if [ ! -x $b ]; then echo "missing $b"; exit 1; fi
    echo "$v $p: $(LD_LIBRARY_PATH=$lib:$BASE_LD ldd $b | grep libdeal_II)"
  done
done

GPU_OPTS="--mpi=pspmix --nodes=$NODES --ntasks=$((4 * NODES)) --ntasks-per-node=4 --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
HYB_OPTS="--mpi=pspmix --nodes=$NODES --ntasks=$((4 * K * NODES)) --ntasks-per-node=$((4 * K)) --cpus-per-task=$((72 / K)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"

i=0
for v in $ORDER; do
  i=$((i + 1))
  lib=$LIB_OLD; [ $v = new ] && lib=$LIB_NEW
  export LD_LIBRARY_PATH=$lib:$BASE_LD

  label=$i-$v-bk3
  echo
  echo "=============== $label: $(binary poisson_cube_bk3 $v) 4 $MIN $MAX 5 5 d 0"
  t0=$SECONDS
  timeout 1800 srun $GPU_OPTS $(binary poisson_cube_bk3 $v) 4 $MIN $MAX 5 5 d 0 2>&1 | tee $OUT/$label.out
  echo "srun elapsed: $((SECONDS - t0)) s"

  label=$i-$v-hybrid-k$K
  echo
  echo "=============== $label: $(binary poisson_cube_hybrid_gpu_cpu $v) $K 4 $MIN $MAX 5 5 d $SWITCH_DOFS 0"
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  t0=$SECONDS
  timeout 2400 srun $HYB_OPTS $(binary poisson_cube_hybrid_gpu_cpu $v) $K 4 $MIN $MAX 5 5 d $SWITCH_DOFS 0 2>&1 | tee $OUT/$label.out
  echo "srun elapsed: $((SECONDS - t0)) s"
done
