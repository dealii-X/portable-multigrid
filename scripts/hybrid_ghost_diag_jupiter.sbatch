#!/bin/bash
#SBATCH --job-name=hybrid-ghost
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=hybrid-ghost-%j.out

# Why is the ghost exchange of the GPU ranks ~18x slower with k >= 12?
# Doubling jobs 2858139..2858145, all levels on the GPUs, fine level of 67.6M DoFs,
# float matvec: computation only 0.75 ms for every k, ghost exchange only 0.18 ms
# at k = 1, 4 and 3.2 - 3.5 ms at k = 12 .. 72.
#
# All runs: Q4, cycle 18 of the doubling mesh (16974593 DoFs), 5 + 5 smoothing,
# switch_dofs = 0 (all levels on the GPUs, the CPU-only ranks idle). Submit from the
# build directory of the hybrid program.
#   A. k = 4 6 8 9 12: where is the step (16, 24, 32, 36, 48 ranks on the node)?
#      Rank 0 prints UCX's protocol table (UCX_PROTO_INFO): which protocol it uses
#      for GPU memory at which message size.
#   B. k = 12 with one setting changed:
#      idle ranks yield the core; UCX told there are 4 processes per node; UCX told
#      there are 4 endpoints. The last two test the guess that UCX picks another
#      protocol when it expects many processes on the node.
# The summary at the end lists the fine-level line of every run.

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

# dealii-neon: NEON on the Grace side, shared Kokkos 5.2.2 (clean exit)
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd ./program | grep -E 'libdeal_II|libkokkoscore'
ucx_info -v
ompi_info --version

FINE=16974593
OUT=ghost-diag-$SLURM_JOB_ID
mkdir -p $OUT

# run <label> <k> [VAR=value ...]: one run, copied to $OUT/<label>.out.
# Rank 0 gets UCX_PROTO_INFO=y, the given variables go to all ranks.
run() {
  local label=$1 k=$2; shift 2
  local common="--mpi=pspmix --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== $label: k=$k $*: ./program $k 4 16000000 17000000 5 5 d 0 0"
  env "$@" timeout 600 srun $common bash -c \
    'if [ "$SLURM_PROCID" = 0 ]; then export UCX_PROTO_INFO=y; fi; exec "$@"' bash \
    ./program $k 4 16000000 17000000 5 5 d 0 0 2>&1 | tee $OUT/$label.out
}

echo "######## A. where is the step"
for k in 4 6 8 9 12; do
  run k$k $k
done

echo "######## B. k=12 with one setting changed"
run k12-yield 12 OMPI_MCA_mpi_yield_when_idle=1
run k12-ppn4 12 UCX_NUM_PPN=4
run k12-eps4 12 UCX_NUM_EPS=4

echo
echo "######## summary: fine level ($FINE DoFs), float matvec and CG solve"
for f in $OUT/*.out; do
  echo "--- $(basename $f .out)"
  grep -E "Best timings for ndof = $FINE   (mv|on level [0-9]+\|  ghost)" $f
done
