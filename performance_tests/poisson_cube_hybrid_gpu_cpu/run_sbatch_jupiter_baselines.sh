#!/bin/bash
#SBATCH --job-name=hybrid-base
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=hybrid-base-%j.out

# Small checks before any big hybrid run, all on the same mesh (ranks per node):
#   1. GPU only    (poisson_cube_bk3)           4 ranks, 1 per GPU
#   2. CPU only    (poisson_cube_cpu)           4 ranks, 1 per Grace
#   3. hybrid regression (all levels on GPUs)   4 ranks (k=1) and 16 ranks (k=4)
#   4. hybrid, CPU-only cycle                   16 ranks (k=4)
# In run 1, rank 0 prints UCX's protocol tables: with several nodes the "inter-node"
# table for messages from cuda memory shows how GPU buffers travel between nodes.
# Mesh: 3x3x3 coarse cells, 3 refinements = 24 cells/dir, Q4 -> 97^3 = 912673 DoFs.
# 5 pre + 5 post Chebyshev smoothing steps everywhere. Mixed precision only: float
# V-cycle under double CG in all three programs (poisson_cube_cpu built without
# -DVCYCLE_DOUBLE; it and the hybrid print their "V-cycle number type").
# CG iterations and reductions of 1, 2 and 3 must agree (CPU may differ in the
# 3rd digit of the reduction: Chebyshev eigenvalue estimates).

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program
CPU_BIN=$REPO/poisson_cube_cpu/$BUILD/program
HYBRID_BIN=$REPO/poisson_cube_hybrid_gpu_cpu/$BUILD/program

# Several nodes: sbatch --nodes=<N> <this script>. Every run then uses all N nodes,
# with 4k ranks on each (k per Grace and GPU).
NODES=${SLURM_JOB_NUM_NODES:-1}

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

ldd $HYBRID_BIN | grep -E 'libdeal_II|libkokkoscore'
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
  local k=$1 bin=$2; shift 2
  local common="--mpi=pspmix --nodes=$NODES --ntasks=$((4 * k * NODES)) --ntasks-per-node=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: $bin $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  timeout 400 srun $common bash -c \
    'if [ "$SLURM_PROCID" = 0 ] && [ -n "$PROTO_INFO" ]; then export UCX_PROTO_INFO=y; fi; exec "$@"' bash \
    $bin "$@"
}

# poisson_cube_{bk3,cpu}: degree minsize maxsize n_pre n_post [s=non-doubling mesh] overlap
# hybrid:                 k degree minsize maxsize n_pre n_post [s] switch_dofs cpu_only_dofs
#                         (both thresholds in DoFs per rank)
# min/max pick the one cycle with 24 cells/dir (sizes 20 and 28 give 531441 and 1442897)
MESH="900000 1000000 5 5 s"

echo "######## 1. GPU only"
PROTO_INFO=1 run 1 $BK3_BIN 4 $MESH 0

echo "######## 2. CPU only"
run 1 $CPU_BIN 4 $MESH 0

echo "######## 3. hybrid, switch_dofs=0, cpu_only_dofs=0 (full GPU hierarchy)"
run 1 $HYBRID_BIN 1 4 $MESH 0 0
run 4 $HYBRID_BIN 4 4 $MESH 0 0

echo "######## 4. hybrid, cpu_only_dofs above the fine level's DoFs per rank (CPU-only cycle)"
run 4 $HYBRID_BIN 4 4 $MESH 0 1000000
