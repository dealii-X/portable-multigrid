#!/bin/bash
#SBATCH --job-name=hybrid-idle
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:45:00
#SBATCH --output=hybrid-idle-%j.out

# Waiting CPU-only ranks: blocking MPI_Bcast (the core spins) against a sleeping wait.
# Busy cores slow the GPU of the same GH200 at full load: GPU-only at 135M DoFs per GPU,
# one fine-level operator application 5.79 ms, with 17 busy cores per Grace 6.06 ms,
# CG solve 614.8 -> 660.7 ms (job 2885286). In the hybrid program with k = 18 the same
# kernel takes 5.90 - 5.96 ms (jobs 2882546, 2884619), with k = 4 5.76 ms.
# Every case k:DoFs runs twice on the same node, with HYBRID_IDLE_SLEEP_US = 0 (as
# before) and 50, each with all levels on the GPUs and as hybrid (switch_dofs = 8000).
# Expected with the sleeping wait: the all-GPU pass as fast as GPU-only.
# Submit from the build directory of the hybrid program (rebuilt with the sleeping wait).

CASES=${CASES:-"18:538970625 36:538970625 36:67634433 18:67634433"}
SLEEP_LIST=${SLEEP_LIST:-"0 50"}

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

OUT=$PWD/hybrid-idle-$SLURM_JOB_ID
mkdir -p $OUT
LABELS=""

for case in $CASES; do
  k=${case%:*}
  dofs=${case#*:}
  for sleep_us in $SLEEP_LIST; do
    label=k$k-$dofs-sleep$sleep_us
    LABELS="$LABELS $label"
    # args after k: degree minsize maxsize n_pre n_post mesh switch_dofs cpu_only_dofs
    args="$k 4 $((dofs - 1000)) $((dofs + 1000)) 5 5 d 0,8000 0"
    echo
    echo "=============== $label: HYBRID_IDLE_SLEEP_US=$sleep_us ./program $args"
    export HYBRID_IDLE_SLEEP_US=$sleep_us
    export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
    t0=$SECONDS
    timeout 1500 srun --mpi=pspmix --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block \
      --cpu-bind=cores --gpu-bind=closest --mem-bind=local ./program $args 2>&1 | tee $OUT/$label.out
    echo "srun elapsed: $((SECONDS - t0)) s"
  done
done

echo
echo "######## summary: CG solve and one operator application on the finest level"
echo "(first pass of each run: all levels on the GPUs; second pass: hybrid)"
for label in $LABELS; do
  f=$OUT/$label.out
  dofs=${label#*-}
  dofs=${dofs%-sleep*}
  echo "--- $label: $(grep -m 1 'Wait of the CPU-only ranks' $f)"
  grep -E "Best timings for ndof = $dofs +mv " $f
  grep -E "Best timings for ndof = $dofs +on level [0-9]+\| +ghost" $f |
    awk '{for (i = 1; i <= NF; i++) if ($i == "level") lev = $(i + 1) + 0; if (lev > max) max = lev; line[NR] = $0; level[NR] = lev}
         END {for (i = 1; i <= NR; i++) if (level[i] == max) print line[i]}'
done
