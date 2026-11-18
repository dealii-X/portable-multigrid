#!/bin/bash
#SBATCH --job-name=hybrid-mps
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=hybrid-mps-%j.out

# Trial of NVIDIA MPS: all processes of a GPU share one CUDA context. Does it remove
# the start-up cost (jobs 2861440..46: MPI + Kokkos initialization ~1.2 s per process,
# 347 s at k = 72), and does it change the solve times?
# Runs k = 72, 4, 1 with MPS on cycle 18 of the doubling mesh (16974593 DoFs), Q4,
# 5 + 5 smoothing, all on GPUs and hybrid with switch_dofs = 8000. Without MPS
# (jobs 2861440..46), CG solve all-GPU / hybrid [ms] and initialization [s]:
#   k = 72: 107.2 / 65.6, 347    k = 4: 87.4 / 65.2, 19    k = 1: 86.8 / 68.4, 7
# Submit from the build directory of the hybrid program.

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

# see run_sbatch_jupiter_doubling.sh
export UCX_NUM_PPN=4

ldd ./program | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader

# MPS control daemon for the four GPUs; the ranks find it through the pipe directory
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
mps_exit=$?
echo "nvidia-cuda-mps-control -d: exit $mps_exit"
if [ $mps_exit -ne 0 ]; then echo "MPS did not start, stopping"; exit 1; fi

run() {
  local k=$1; shift
  local common="--mpi=pspmix --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== MPS, k=$k: ./program $k $*"
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 1200 srun $common ./program $k "$@"
  echo "srun elapsed: $((SECONDS - t0)) s"
  # GPU memory in use after the run (the MPS servers stay up between runs)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

# args after k: degree minsize maxsize n_pre n_post mesh switch_dofs cpu_only_dofs
for k in 72 4 1; do
  run $k 4 16000000 17000000 5 5 d 0,8000 0
done

echo
echo "######## MPS shutdown and logs"
echo quit | nvidia-cuda-mps-control
cat $CUDA_MPS_LOG_DIRECTORY/control.log
