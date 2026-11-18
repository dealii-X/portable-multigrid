#!/bin/bash
#SBATCH --job-name=hybrid-sharing
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --output=hybrid-sharing-%j.out

# Why do the GPU kernels of the fine levels run slower with many ranks per GPU?
# 135M DoFs per GPU, one node, one operator application on the finest level [ms]
# (jobs 2882551, 2882892, 2882546):
#                 computation   ghost exchange
#   GPU-only         5.72           0.46
#   hybrid k = 4     5.76           0.43
#   hybrid k = 18    5.96           0.46
# The loss is in the computation and grows with the kernel time, so it is not the
# constant cost per call of busy neighbouring cores (job 2880860). Suspect: the GPU is
# shared by the K processes of its Grace, each with its own CUDA context, and the
# waiting CPU-only ranks touch it while polling for messages.
# Three runs with K ranks per GPU on the same node, Q4, 5 + 5 smoothing, 538970625 DoFs,
# all levels on the GPUs and hybrid with switch_dofs = 8000:
#   A. as in the other jobs (reference on this node)
#   B. with NVIDIA MPS: all processes of a GPU share one context
#   C. without MPS, CPU-only ranks without the CUDA transports of UCX
# B as fast as k = 4: sharing of the GPU between contexts is the cause.
# C as fast as k = 4: the polling of the CPU-only ranks reaches the GPU through UCX.
# Submit from the build directory of the hybrid program.

K=${K:-18}
DOFS=${DOFS:-538970625}

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

OUT=$PWD/hybrid-sharing-$SLURM_JOB_ID
mkdir -p $OUT

# args after k: degree minsize maxsize n_pre n_post mesh switch_dofs cpu_only_dofs
ARGS="$K 4 $((DOFS - 1000)) $((DOFS + 1000)) 5 5 d 0,8000 0"
COMMON="--mpi=pspmix --ntasks=$((4 * K)) --cpus-per-task=$((72 / K)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"

# run <label> <wrapper ...>: one run, copied to $OUT/<label>.out. The wrapper, if any,
# is put in front of the program on every rank.
run() {
  local label=$1; shift
  echo
  echo "=============== $label: ./program $ARGS"
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 1500 srun $COMMON "$@" ./program $ARGS 2>&1 | tee $OUT/$label.out
  echo "srun elapsed: $((SECONDS - t0)) s"
}

echo "######## A. reference: K = $K, no MPS"
run A-reference

echo "######## B. K = $K with MPS"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
mps_exit=$?
echo "nvidia-cuda-mps-control -d: exit $mps_exit"
if [ $mps_exit -eq 0 ]; then
  run B-mps
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo quit | nvidia-cuda-mps-control
  tail -n 20 $CUDA_MPS_LOG_DIRECTORY/control.log
else
  echo "MPS did not start, run B skipped"
fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

echo "######## C. K = $K, no MPS, CPU-only ranks without the CUDA transports of UCX"
# the GPU rank is the first rank of each block of K
run C-no-ucx-cuda bash -c 'if [ $((SLURM_PROCID % '$K')) -ne 0 ]; then export UCX_TLS=^cuda; fi; exec "$@"' bash

echo
echo "######## summary: CG solve and one operator application on the finest level"
echo "(first pass of each run: all levels on the GPUs; second pass: hybrid)"
for l in A-reference B-mps C-no-ucx-cuda; do
  [ -f $OUT/$l.out ] || continue
  echo "--- $l"
  grep -E "Best timings for ndof = $DOFS +mv " $OUT/$l.out
  grep -E "Best timings for ndof = $DOFS +on level [0-9]+\| +ghost" $OUT/$l.out |
    awk '{for (i = 1; i <= NF; i++) if ($i == "level") lev = $(i + 1) + 0; if (lev > max) max = lev; line[NR] = $0; level[NR] = lev}
         END {for (i = 1; i <= NR; i++) if (level[i] == max) print line[i]}'
done
