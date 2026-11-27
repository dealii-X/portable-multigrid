#!/bin/bash
#SBATCH --job-name=elasticity-vmult
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=elasticity-vmult-%j.out

# Timings of the vmult variants of the elasticity operator
# (performance_tests/elasticity_operator: vmult_dealii, vmult_batched,
# vmult_fused, vmult_benchmark), degree + 1 points per direction, 4 ranks with
# one H100 each, on doubling meshes from 8M DoFs until a run fails (out of
# memory) or times out: the largest size per GPU is not known yet. One size per
# program run; the GPU memory is sampled every 2 s and its peak printed per run.
#
#   DEGREES="1 2 3 4"  degrees to run, split into several jobs if needed
#   START_DOFS=8e6     lower end of the ladder (total DoFs), e.g. to resume
#   MAX_DOFS=4.2e9     upper end of the ladder (total DoFs)
#   RUN_TIMEOUT=1800   seconds per program run (an OOM can hang the ranks)
#   REPS=20            repetitions per variant
#   BATCHES="0"        cells per batch for all but vmult_dealii, one program
#                      run each; 0 = the benchmark's rule
#   BUILD=$HOME/codes/dealiiX-elasticity/build-elasticity-operator
# A run whose srun fails with a SLURM communication error ("Unable to confirm
# allocation", "Socket timed out") is retried up to 3 times, 60 s apart.

BUILD=${BUILD:-$HOME/codes/dealiiX-elasticity/build-elasticity-operator}
DEGREES=${DEGREES:-"1 2 3 4"}
START_DOFS=${START_DOFS:-8e6}
MAX_DOFS=${MAX_DOFS:-4.2e9}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
REPS=${REPS:-20}
BATCHES=${BATCHES:-0}

cd "$(dirname "$BUILD")"
source scripts/jupiter/env.sh "${DEAL_II_DIR:-$WORK/dealii/dealii-master}"
export UCX_NUM_PPN=4

PROGRAM=$BUILD/program
ldd $PROGRAM | grep -E 'libdeal_II|libkokkoscore'

ERRORS=$(mktemp)
SAMPLES=$(mktemp)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > "$SAMPLES" &
SAMPLER=$!

summary=""
for degree in $DEGREES; do
  dofs=$START_DOFS
  while python3 -c "import sys; sys.exit(0 if $dofs <= $MAX_DOFS else 1)"; do
    failed=0
    for batch in $BATCHES; do
      start_line=$(wc -l < "$SAMPLES")
      echo
      echo "=============== degree $degree, target $dofs DoFs, cells per batch $batch"
      for attempt in 1 2 3; do
        timeout $RUN_TIMEOUT srun --mpi=pspmix --ntasks=4 --cpus-per-task=72 --cpu-bind=cores \
          --gpu-bind=closest $PROGRAM $degree $dofs $REPS $batch 2> "$ERRORS"
        status=$?
        cat "$ERRORS" >&2
        if [ $status -ne 0 ] && grep -q -e "Unable to confirm allocation" -e "Socket timed out" "$ERRORS"; then
          echo "SLURM communication error (attempt $attempt), retrying in 60 s"
          sleep 60
          continue
        fi
        break
      done
      peak=$(tail -n +$((start_line + 1)) "$SAMPLES" | awk -F', ' '$2 > max { max = $2 } END { print max + 0 }')
      echo "peak GPU memory during the run: $peak MiB (of 97871)"
      summary="$summary\ndegree $degree target $dofs batch $batch: exit $status, peak $peak MiB"
      [ $status -ne 0 ] && failed=1
    done
    if [ $failed -ne 0 ]; then
      echo "degree $degree: run failed or timed out at $dofs DoFs, next degree"
      break
    fi
    dofs=$(python3 -c "print(2 * $dofs)")
  done
done

kill $SAMPLER
rm -f "$SAMPLES" "$ERRORS"

echo
echo "=============== summary"
echo -e "$summary"
