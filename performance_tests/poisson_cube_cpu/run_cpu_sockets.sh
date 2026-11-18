#!/bin/bash
#SBATCH --job-name=cpu-sockets
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:30:00
#SBATCH --output=cpu-sockets-%j.out

# Repeat the CPU runs of run_gpu_vs_cpu.sh with the ranks packed onto whole sockets
# (--distribution=block:block) instead of Slurm's default cyclic placement over the
# four Grace sockets. That also matters at 288 cores: with the default, neighbouring
# subdomains sit on different Graces and the CG solve is 3 - 5 ms slower at every
# size (job 2861438). Same labels, so plot_gpu_vs_cpu.py can overlay them on the
# main job: plot_gpu_vs_cpu.py <main jobid> . <this jobid>

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

CPU32=build-neon/program
CPU64=build-neon-fp64/program
mkdir -p results

SRUN_CPU="srun --mpi=pspmix --distribution=block:block --cpu-bind=cores --mem-bind=local --cpus-per-task=1"

echo "===== placement check: first and last ranks of a 72-task step ====="
$SRUN_CPU --ntasks=72 bash -c 'echo "rank $SLURM_PROCID $(grep Cpus_allowed_list /proc/self/status) socket $(cat /sys/devices/system/cpu/cpu$(grep -o "Cpus_allowed_list:\s*[0-9]*" /proc/self/status | grep -o "[0-9]*$")/topology/physical_package_id)"' | sort -k2 -n | sed -n '1,3p;70,72p'
echo "===== placement check: 144 tasks, one line per socket id ====="
$SRUN_CPU --ntasks=144 bash -c 'cat /sys/devices/system/cpu/cpu$(grep -o "Cpus_allowed_list:\s*[0-9]*" /proc/self/status | grep -o "[0-9]*$")/topology/physical_package_id' | sort | uniq -c

echo "===== launch check ====="
$SRUN_CPU --ntasks=4 $CPU32 4 1 1000 3 3 d 0 > results/launch_check.$SLURM_JOB_ID.out 2>&1
echo "launch exit=$?"
grep -m1 "Number of MPI ranks" results/launch_check.$SLURM_JOB_ID.out
grep -q "Number of MPI ranks:                   4" results/launch_check.$SLURM_JOB_ID.out || { echo "MPI launch failed (ranks != 4), aborting"; exit 1; }

run() {
  local label=$1 sr=$2 prog=$3 maxsize=$4; shift 4
  echo "===== $label ====="
  $sr $prog 4 1 $maxsize 5 5 d "$@" > results/$label.$SLURM_JOB_ID.out 2>&1
  echo "exit=$?"
  grep -m1 -E "Number of MPI ranks" results/$label.$SLURM_JOB_ID.out
  grep -m1 -E "V-cycle number type" results/$label.$SLURM_JOB_ID.out
  grep -E "^Best timings for ndof = [0-9]+ +mv" results/$label.$SLURM_JOB_ID.out | tail -3
}

run cpu64.np72     "$SRUN_CPU --ntasks=72"  $CPU64 100000000 0
run cpu64.np144    "$SRUN_CPU --ntasks=144" $CPU64 200000000 0
run cpu64.np288    "$SRUN_CPU --ntasks=288" $CPU64 400000000 0
run cpu32.np72     "$SRUN_CPU --ntasks=72"  $CPU32 100000000 0
run cpu32.np144    "$SRUN_CPU --ntasks=144" $CPU32 200000000 0
run cpu32.np288    "$SRUN_CPU --ntasks=288" $CPU32 400000000 0

echo "===== done ====="
