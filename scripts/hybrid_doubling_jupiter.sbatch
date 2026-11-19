#!/bin/bash
#SBATCH --job-name=hybrid-dbl
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --output=hybrid-dbl-%j.out

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

# dealii-neon: NEON on the Grace side, shared Kokkos 5.2.2 (clean exit)
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

# With 32 or more processes on the node UCX 1.19 stops using cuda_ipc for messages
# from GPU memory and stages them through host shared memory: the ghost exchange of
# the GPU ranks gets ~10x slower (job 2859067). Tell it there are 4 per node.
export UCX_NUM_PPN=4

ldd ./program | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

# n ranks, k per Grace (= per GPU), 4k ranks on the node. Each rank gets
# 72/k cores; block placement then puts ranks 0..k-1 on the cores of Grace 0,
# k..2k-1 on Grace 1, ... (needs 72 % k == 0). --distribution=block:block is
# needed: the default is cyclic over sockets, which puts consecutive ranks on
# different Graces and all GPU ranks on GPU 0 (k=4 baselines run: 36x
# slower). --ntasks-per-socket must NOT be combined with it: the step then got
# cores of one Grace only and ranks were left unpinned (~50 ms MPI latency).
run() {
  local k=$1; shift
  if [ $((72 % k)) -ne 0 ]; then echo "k=$k does not divide the 72 cores of a Grace, skipped"; return; fi
  local common="--mpi=pspmix --nodes=$NODES --ntasks=$((4 * k * NODES)) --ntasks-per-node=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: ./program $k $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  # start-up: the program prints the time from here to main() and its MPI + Kokkos
  # initialization time
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 6000 srun $common ./program $k "$@"
  echo "srun elapsed: $((SECONDS - t0)) s"
}

# The usual doubling-mesh test: Q4, 5 pre + 5 post Chebyshev steps, cycles 12..20 of
# poisson_cube_bk3's sequence (274625 .. 67M DoFs). One program run per k goes
# through the cycles once per pair of thresholds, so start-up is paid once.
# Compare with the GPU-only and CPU-only runs
#   ./program 4 200000 70000000 5 5 d 0
# args after k: degree minsize maxsize n_pre n_post mesh switch_dofs cpu_only_dofs
#
# Both thresholds are DoFs per rank (4k ranks on the node), comma-separated lists
# of the same length. switch_dofs: the switch level is the finest level below the
# fine one with at most that many (0 = all levels on the GPUs). cpu_only_dofs: a
# cycle whose fine level has at most that many runs on the CPUs only (0 = never).
# Default: all on GPUs; hybrid with 2000; hybrid with 8000; hybrid with 8000 and
# CPU-only up to 150000 per rank, to find where CPU-only stops paying off (rounds
# 1 and 2 only reached 30000).
# k has to divide 72 (72/k cores per rank): 4k = 4 .. 288 ranks on the node.
# Override with K_LIST="..." SWITCH_DOFS="..." CPU_ONLY_DOFS="..." sbatch. One job per k:
#   for k in 1 4 12 18 24 36 72; do K_LIST=$k sbatch --job-name=hybrid-dbl-k$k <this script>; done
K_LIST=${K_LIST:-"1 4 12 18 24 36 72"}
SWITCH_DOFS=${SWITCH_DOFS:-"0,2000,8000,8000"}
CPU_ONLY_DOFS=${CPU_ONLY_DOFS:-"0,0,0,150000"}

# Several nodes: sbatch --nodes=<N> <this script>. Every run then uses all N nodes,
# with 4k ranks on each (k per Grace and GPU).
NODES=${SLURM_JOB_NUM_NODES:-1}
# The size range grows with the node count: 2 nodes run the cycles 13..21, each
# with the DoFs per rank of the cycle before it on 1 node (weak scaling).
# MIN_PER_NODE / MAX_PER_NODE (DoFs) change the range, e.g. for the large cycles up
# to 67M DoFs per GPU, where no cycle is CPU-only any more:
#   MIN_PER_NODE=60000000 MAX_PER_NODE=280000000 SWITCH_DOFS=0,2000,8000 CPU_ONLY_DOFS=0
MIN_PER_NODE=${MIN_PER_NODE:-200000}
MAX_PER_NODE=${MAX_PER_NODE:-70000000}
for k in $K_LIST; do
  run $k 4 $((MIN_PER_NODE * NODES)) $((MAX_PER_NODE * NODES)) 5 5 d $SWITCH_DOFS $CPU_ONLY_DOFS
done
