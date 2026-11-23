#!/bin/bash
#SBATCH --job-name=mg-transfer-ab
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=mg-transfer-ab-%j.out

# "Setup transfers" before and after the geometric transfer takes its data from
# deal.II's host transfer: poisson_cube_bk3 of main (OLD_BIN) against the one of
# this branch (NEW_BIN), same node, doubling mesh, 5 + 5 smoothing, 1 rank per
# GPU. Full runs, so the CG table checks the result as well: cg_its must agree.
# Sizes per GPU: MIN_PER_GPU .. 270M for p=4, .. 135M for p=2 (270M does not
# fit for p=2, job 3077994; whether p=2 solves at 135M is not known).
#   REPO=<checkout of this branch> OLD_BIN=<main build>/poisson_cube_bk3 \
#   NEW_BIN=<this build>/poisson_cube_bk3 [DEAL_II_DIR=<install>] sbatch <this script>
# RUNS: "<degree>:<GPUs>" list, the important ones first.

REPO=${REPO:?set REPO to the checkout of this branch}
OLD_BIN=${OLD_BIN:?set OLD_BIN to poisson_cube_bk3 built from main}
NEW_BIN=${NEW_BIN:?set NEW_BIN to poisson_cube_bk3 built from this branch}
RUNS=${RUNS:-"4:4 4:1 2:4 2:1"}
MIN_PER_GPU=${MIN_PER_GPU:-8000000}
MAX_PER_GPU_P4=${MAX_PER_GPU_P4:-271000000}
MAX_PER_GPU_P2=${MAX_PER_GPU_P2:-136000000}

source $REPO/scripts/jupiter/env.sh $DEAL_II_DIR
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $NEW_BIN | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OUT=$PWD/mg-transfer-ab-$SLURM_JOB_ID
mkdir -p $OUT

for run in $RUNS; do
  p=${run%:*}
  gpus=${run#*:}
  if [ $p = 2 ]; then max=$MAX_PER_GPU_P2; else max=$MAX_PER_GPU_P4; fi
  opts="--mpi=pspmix --ntasks=$gpus --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  for version in new old; do
    if [ $version = new ]; then bin=$NEW_BIN; else bin=$OLD_BIN; fi
    label=p${p}_gpus${gpus}_$version
    echo
    echo "=============== $label: srun $opts $bin $p $((MIN_PER_GPU * gpus)) $((max * gpus)) 5 5 d 0"
    timeout 3000 srun $opts $bin $p $((MIN_PER_GPU * gpus)) $((max * gpus)) 5 5 d 0 > $OUT/$label.out 2>&1
    echo "exit code $?"
    grep -E "Number of MPI ranks|failed to allocate|Kokkos ERROR" $OUT/$label.out
  done

  echo
  echo "######## p=$p, $gpus GPU(s): Setup transfers [s] and CG, old | new"
  for version in old new; do
    f=$OUT/p${p}_gpus${gpus}_$version.out
    # per cycle: DoFs and transfer setup time
    awk '/^Number of degrees of freedom:/ {n = $6}
         /^Setup transfers/ {t = $NF; sub(/.*s\//, "", t); sub(/s$/, "", t); print n, t}' $f > $OUT/setup_$version.txt
    # last CG table: dofs, cg_its, cg_reduction
    awk '/cg_its/ {for (i = 1; i <= NF; i++) {if ($i == "dofs") d = i; if ($i == "cg_its") c = i; if ($i == "cg_reduction") r = i}; h = NF; n = 0; next}
         d && NF == h && $d ~ /^[0-9]+$/ && $c ~ /^[0-9]+$/ {rows[n++] = $d " " $c " " $r}
         END {for (i = 0; i < n; i++) print rows[i]}' $f > $OUT/cg_$version.txt
  done
  join <(sort $OUT/setup_old.txt) <(sort $OUT/setup_new.txt) | sort -n | awk '{printf "setup  %12d  old %9.3f  new %9.3f  old/new %5.2f\n", $1, $2, $3, $2 / $3}'
  join <(sort $OUT/cg_old.txt) <(sort $OUT/cg_new.txt) | sort -n | awk '{printf "cg     %12d  its %s | %s  reduction %s | %s%s\n", $1, $2, $4, $3, $5, ($2 == $4 ? "" : "   <-- ITERATIONS DIFFER")}'
done
