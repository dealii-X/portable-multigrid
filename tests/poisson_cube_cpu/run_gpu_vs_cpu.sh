#!/bin/bash
#SBATCH --job-name=gpu-vs-cpu
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=gpu-vs-cpu-%j.out

# GPU multigrid (tests/poisson_cube_bk3 = float V-cycle under double CG, and its
# fp64 twin, 1 rank per GH200 GPU) against CPU multigrid
# (tests/poisson_cube_cpu, 1 rank per Grace core), same deal.II install, same node,
# 5+5 Chebyshev smoothing steps, 1, 2 and 4 GPUs against 72, 144 and 288 cores.
# Run from tests/poisson_cube_cpu of a gpu-vs-cpu checkout on JUPITER; the GPU programs
# are built in ../poisson_cube_bk3/build-neon and ../poisson_cube_bk3_fp64/build-neon.

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export CUDA_DEVICE_WAITS_ON_EXCEPTION=0
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

GPU32=../poisson_cube_bk3/build-neon/program
GPU64=../poisson_cube_bk3_fp64/build-neon/program
CPU32=build-neon/program
CPU64=build-neon-fp64/program
mkdir -p results

echo "===== node ====="
hostname; nproc; nvidia-smi -L
for p in $GPU32 $GPU64 $CPU32 $CPU64; do echo "--- $p"; ldd $p | grep -E "libdeal_II|libkokkoscore"; done

SRUN_GPU="srun --mpi=pspmix --cpu-bind=cores --gpu-bind=closest --mem-bind=local --gpus-per-task=1 --cpus-per-task=72"
# --distribution=block:block: the default spreads consecutive ranks over the four
# Graces (rank 0 on core 0, rank 1 on core 72, ...), also at 288 ranks; neighbouring
# subdomains then exchange ghosts across sockets and the CG solve is 3 - 5 ms slower
# at every size (job 2861438). The CPU runs of job 2843214 were launched without it.
SRUN_CPU="srun --mpi=pspmix --distribution=block:block --cpu-bind=cores --mem-bind=local --cpus-per-task=1"

echo "===== launch check ====="
$SRUN_CPU --ntasks=4 $CPU32 4 1 1000 3 3 d 0 > results/launch_check.$SLURM_JOB_ID.out 2>&1
echo "launch exit=$?"
grep -m1 "Number of MPI ranks" results/launch_check.$SLURM_JOB_ID.out
grep -q "Number of MPI ranks:                   4" results/launch_check.$SLURM_JOB_ID.out || { echo "MPI launch failed (ranks != 4), aborting"; exit 1; }

run() { # label, srun line, program, maxsize, extra args
  local label=$1 sr=$2 prog=$3 maxsize=$4; shift 4
  echo "===== $label ====="
  $sr $prog 4 1 $maxsize 5 5 d "$@" > results/$label.$SLURM_JOB_ID.out 2>&1
  echo "exit=$?"
  grep -m1 -E "Number of MPI ranks" results/$label.$SLURM_JOB_ID.out
  grep -m1 -E "V-cycle number type" results/$label.$SLURM_JOB_ID.out
  grep -E "^Best timings for ndof = [0-9]+ +mv" results/$label.$SLURM_JOB_ID.out | tail -3
}

# 5+5 smoothing steps; max sizes keep ~150M DoFs per GPU and ~1.4M DoFs per core.
# bk3 and the CPU program take the overlap flag as 7th argument, bk3_fp64 does not.
run gpu64.np1      "$SRUN_GPU --ntasks=1"   $GPU64 150000000
run gpu64.np2      "$SRUN_GPU --ntasks=2"   $GPU64 300000000
run gpu64.np4      "$SRUN_GPU --ntasks=4"   $GPU64 600000000
run cpu64.np72     "$SRUN_CPU --ntasks=72"  $CPU64 100000000 0
run cpu64.np144    "$SRUN_CPU --ntasks=144" $CPU64 200000000 0
run cpu64.np288    "$SRUN_CPU --ntasks=288" $CPU64 400000000 0
run gpu32.np1      "$SRUN_GPU --ntasks=1"   $GPU32 150000000 0
run gpu32.np2      "$SRUN_GPU --ntasks=2"   $GPU32 300000000 0
run gpu32.np4      "$SRUN_GPU --ntasks=4"   $GPU32 600000000 0
run cpu32.np72     "$SRUN_CPU --ntasks=72"  $CPU32 100000000 0
run cpu32.np144    "$SRUN_CPU --ntasks=144" $CPU32 200000000 0
run cpu32.np288    "$SRUN_CPU --ntasks=288" $CPU32 400000000 0

echo "===== done ====="
