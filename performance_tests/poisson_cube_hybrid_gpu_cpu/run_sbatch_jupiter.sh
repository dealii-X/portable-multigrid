#!/bin/bash
#SBATCH --job-name=hybrid-sweep
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=hybrid-%j.out

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
  local common="--mpi=pspmix --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: ./program $k $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  # start-up: the program prints the time from here to main() and its MPI + Kokkos
  # initialization time
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 2400 srun $common ./program $k "$@"
  echo "srun elapsed: $((SECONDS - t0)) s"
}

# args after k: degree minsize maxsize n_pre n_post mesh switch_dofs cpu_only_dofs
# Q4, 24 cells/dir (912673 DoFs). Levels and their DoFs: 0..3 = Q1 h-levels (64, 343,
# 2197, 15625), 4 = Q2 (117649), 5 = Q4 (fine, 912673). One program run per k with
# six pairs of thresholds (DoFs per rank): switch_dofs selects the levels 0..4 in
# turn (0 = full GPU hierarchy, regression) with cpu_only_dofs = 0, the last pair
# runs the cycle on the CPUs only.
# 5 pre + 5 post smoothing steps (the October sweep, job 2320320, used 3 + 3).
# k has to divide 72 (72/k cores per rank): 4k = 4 .. 288 ranks on the node.
for k in 1 4 12 18 24 36 72; do
  ranks=$((4 * k))
  switch_dofs=0
  for level_dofs in 343 2197 15625 117649; do
    switch_dofs=$switch_dofs,$(((level_dofs + ranks - 1) / ranks))
  done
  run $k 4 900000 1000000 5 5 s $switch_dofs,0 0,0,0,0,0,1000000
done
