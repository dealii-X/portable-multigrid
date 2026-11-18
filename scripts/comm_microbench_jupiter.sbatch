#!/bin/bash
#SBATCH --job-name=comm-bench
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=comm-bench-%j.out

# Phase 2B of tests/poisson_cube_hybrid_gpu_cpu/PLAN.md. Submit from the build dir of
# tests/comm_microbench (binary ./program). CSV lines: test,mem,idle,nranks,group,...
# Rank layout as in the hybrid runs: k ranks per Grace, ranks 0..k-1 on Grace 0, ...;
# the "GPU ranks" are 0, k, 2k, 3k. Compare the same pair in jobs with 4, 16 and 72 ranks
# to see the effect of extra (idle) ranks.

module purge
module load Stages/2026
module load nvidia-compilers
module load OpenMPI
module load FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-mg-transfer/lib:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -c 0

ldd ./program | grep libdeal_II
ompi_info 2>/dev/null | grep -i -E "cuda|ucx" | head
ucx_info -v 2>/dev/null | head -3
nvidia-smi topo -m 2>/dev/null | head -8

MAXB=16777216
REPS=200

bench() {  # k, then program arguments
  local k=$1; shift
  echo "### k=$k: $*"
  timeout 300 srun --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block \
       --cpu-bind=cores --gpu-bind=closest --mem-bind=local ./program "$@" \
       --max-bytes $MAXB --reps $REPS 2> >(grep -vE '^\[[^]]+\] \[ *[0-9]+\]|End of error message' >&2)
  wait
}

for k in 1 4 18; do
  g="0,$k,$((2 * k)),$((3 * k))"            # the four GPU ranks
  for mem in host staged device; do
    # between GPU ranks of different Graces (and, for k > 1, inside one Grace)
    bench $k pingpong --group 0,$k --mem $mem
    [ $k -gt 1 ] && bench $k pingpong --group 0,1 --mem $mem
    bench $k halo --group $g --mem $mem
  done
  # does it matter how the idle ranks wait?
  [ $k -gt 1 ] && bench $k pingpong --group 0,$k --mem staged --idle sleep
  [ $k -gt 1 ] && bench $k halo --group $g --mem staged --idle sleep
  # CG dot products: allreduce over the GPU ranks and over everybody
  bench $k allreduce --group $g
  bench $k allreduce
done
