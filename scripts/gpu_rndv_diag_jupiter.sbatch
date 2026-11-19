#!/bin/bash
#SBATCH --job-name=gpu-rndv
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=gpu-rndv-%j.out

# Can the ghost exchange of the GPU ranks between nodes be made cheaper?
# Baselines on 1 / 2 / 4 nodes (jobs 2857956, 2864278, 2864912): fine-level float
# matvec, ghost exchange only 0.123 / 0.218 / 0.323 ms, computation 0.055 ms. Between
# nodes UCX sends everything from 3563 bytes on from GPU memory by rendezvous
# (zero-copy RDMA), which costs a handshake per message.
# poisson_cube_bk3, 1 rank per GPU, Q4, 5 + 5 smoothing, one cycle of the doubling
# mesh with 4.2M DoFs per GPU, with the rendezvous threshold raised step by step.
# Rank 0 prints UCX's protocol tables, the summary lists the ghost-exchange time of
# every level. Other node counts: sbatch --nodes=<N>.

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
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

ldd $BK3_BIN | grep -E 'libdeal_II|libkokkoscore'

OUT=gpu-rndv-$SLURM_JOB_ID
mkdir -p $OUT
ARGS="4 $((16000000 * NODES)) $((17500000 * NODES)) 5 5 d 0"

# run <label> [VAR=value ...]: one run, copied to $OUT/<label>.out
run() {
  local label=$1; shift
  echo
  echo "=============== $label: $*: $BK3_BIN $ARGS"
  env "$@" timeout 400 srun --mpi=pspmix --nodes=$NODES --ntasks=$((4 * NODES)) --ntasks-per-node=4 --cpus-per-task=72 \
    --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local bash -c \
    'if [ "$SLURM_PROCID" = 0 ]; then export UCX_PROTO_INFO=y; fi; exec "$@"' bash \
    $BK3_BIN $ARGS 2>&1 | tee $OUT/$label.out
}

run default
run rndv-64k UCX_RNDV_THRESH=65536
run rndv-1m UCX_RNDV_THRESH=1048576
run rndv-inf UCX_RNDV_THRESH=inf

echo
echo "######## summary: CG solve and ghost exchange per level"
for f in $OUT/default.out $OUT/rndv-64k.out $OUT/rndv-1m.out $OUT/rndv-inf.out; do
  echo "--- $(basename $f .out)"
  grep -E "Best timings for ndof = [0-9]+   (mv|on level [0-9]+\|  ghost)" $f
done
