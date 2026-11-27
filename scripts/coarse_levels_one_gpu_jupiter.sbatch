#!/bin/bash
#SBATCH --job-name=coarse-one-gpu
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=coarse-one-gpu-%j.out

# poisson_cube_bk3 (performance_tests/poisson_cube_multigrid) with the h-levels of
# fewer than N cells on rank 0 (--single-rank-cells N), for every N in THRESHOLDS.
# N = 0 is the reference (16 cells per rank at least). 4 GPUs per node, 1 rank per GPU:
#   sbatch <this script>                      4 GPUs
#   sbatch --nodes=4 <this script>            16 GPUs
#   THRESHOLDS="0 40000" REPS=2 MAX_PER_GPU=135000000 sbatch --time=03:00:00 <this script>
# Doubling mesh, 5 + 5 smoothing, float V-cycle, MIN_PER_GPU .. MAX_PER_GPU DoFs.
# The cells of an h-level are (1, 2 or 4) * 8^l, so the default thresholds move one
# more level each. One JSON per run in OUT; at the end a table of cg_time against
# N = 0 (cg_its must agree).

# checkout and binary, override with REPO=... BIN=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-coarse-levels-one-gpu}
BIN=${BIN:-$REPO/build/performance_tests/poisson_cube_multigrid/poisson_cube_bk3}

NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS=$((4 * NODES))
DEGREE=${DEGREE:-4}
THRESHOLDS=${THRESHOLDS:-"0 5000 40000 300000 2500000"}
REPS=${REPS:-1}
MIN_PER_GPU=${MIN_PER_GPU:-250000}
MAX_PER_GPU=${MAX_PER_GPU:-34000000}
MIN=$((MIN_PER_GPU * GPUS))
MAX=$((MAX_PER_GPU * GPUS))
TIMEOUT=${TIMEOUT:-1200} # seconds per run
OUT=${OUT:-$PWD/coarse-one-gpu-${SLURM_JOB_ID:-local}}

source $REPO/scripts/jupiter/env.sh
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

mkdir -p $OUT
ldd $BIN | grep -E 'libdeal_II|libkokkoscore'
echo "GPUs: $GPUS, sizes $MIN .. $MAX DoFs, degree: $DEGREE, thresholds: $THRESHOLDS, repetitions: $REPS"
echo "JSON files: $OUT"

opts="--mpi=pspmix --nodes=$NODES --ntasks=$GPUS --ntasks-per-node=4 --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"

echo "placement:"
srun $opts bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V

for rep in $(seq 1 $REPS); do
  for n in $THRESHOLDS; do
    echo
    echo "=============== single-rank-cells=$n repetition=$rep"
    timeout $TIMEOUT srun $opts $BIN $DEGREE $MIN $MAX 5 5 d 0 \
      --single-rank-cells $n --json $OUT/n${n}_r${rep}.json
  done
done

echo
echo "=============== cg_time [s] and its ratio to N = 0, GPUs: $GPUS, degree: $DEGREE"
python3 - $OUT $REPS $THRESHOLDS << 'EOF'
import json, os, sys

out, reps, thresholds = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

for rep in range(1, reps + 1):
    runs = {}
    for n in thresholds:
        name = os.path.join(out, "n%s_r%d.json" % (n, rep))
        if os.path.isfile(name):
            with open(name) as f:
                runs[n] = {c["dofs"]: c for c in json.load(f)["cycles"]}
    if "0" not in runs:
        print("repetition %d: no run with N = 0" % rep)
        continue

    print("repetition %d" % rep)
    print("%12s %4s" % ("dofs", "its") + "".join("%20s" % ("N=" + n) for n in runs))
    for dofs, ref in sorted(runs["0"].items()):
        line = "%12d %4d" % (dofs, ref["cg_its"])
        for n, cycles in runs.items():
            if dofs not in cycles:
                line += "%20s" % "-"
                continue
            c = cycles[dofs]
            flag = "" if c["cg_its"] == ref["cg_its"] else " its=%d!" % c["cg_its"]
            line += "%20s" % ("%.3e (%.2f)%s" % (c["cg_time"], c["cg_time"] / ref["cg_time"], flag))
        print(line)
EOF
