#!/bin/bash
#SBATCH --job-name=coarse-vs-hybrid
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=coarse-vs-hybrid-%j.out

# Coarse levels on one GPU against the hybrid CPU-GPU multigrid, same nodes, same meshes:
#   A  poisson_cube_bk3 --single-rank-cells 0           reference
#   B  poisson_cube_bk3 --single-rank-cells N_CELLS     h-levels below N_CELLS cells on GPU 0
#   C  hybrid, K ranks per GPU, switch SWITCH_DOFS      DoFs per rank, its standard setting
#   D  hybrid, K ranks per GPU, switch MATCH_DOFS / ranks: levels up to MATCH_DOFS DoFs on
#      the CPUs, the levels of B where the fine mesh has more than N_CELLS cells
# C also moves the Q1 / Q2 levels of the fine mesh where they are below its limit; B cannot.
# 4 GPUs per node:
#   sbatch <this script>                      4 GPUs
#   sbatch --nodes=4 <this script>            16 GPUs
#   K=36 RUNS="C D" REPS=1 sbatch <this script>
# p = 4, doubling mesh, 5 + 5 smoothing, float V-cycle, MIN_PER_GPU .. MAX_PER_GPU DoFs.
# One log per run in OUT; at the end a table of cg_time against A.

# checkout, deal.II install and binaries, override with REPO=... MG_BIN=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-coarse-levels-one-gpu}
DEAL_II=${DEAL_II:-/e/project1/dealii-x/ivan/sw/dealii/dealii-neon}
MG_BIN=${MG_BIN:-$REPO/performance_tests/poisson_cube_multigrid/build-neon/poisson_cube_bk3}
HYBRID_BIN=${HYBRID_BIN:-$REPO/performance_tests/poisson_cube_hybrid_gpu_cpu/build-neon/program}

NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS=$((4 * NODES))
K=${K:-18}
RUNS=${RUNS:-"A B C D"}
REPS=${REPS:-2}
N_CELLS=${N_CELLS:-300000}
SWITCH_DOFS=${SWITCH_DOFS:-8000}
MATCH_DOFS=${MATCH_DOFS:-300000}
MATCH_SWITCH=$((MATCH_DOFS / (GPUS * K)))
MIN_PER_GPU=${MIN_PER_GPU:-250000}
MAX_PER_GPU=${MAX_PER_GPU:-34000000}
MIN=$((MIN_PER_GPU * GPUS))
MAX=$((MAX_PER_GPU * GPUS))
TIMEOUT=${TIMEOUT:-2400} # seconds per run
OUT=${OUT:-$PWD/coarse-vs-hybrid-${SLURM_JOB_ID:-local}}

if [ $((72 % K)) -ne 0 ]; then echo "K=$K does not divide the 72 cores of a Grace"; exit 1; fi

source $REPO/scripts/jupiter/env.sh $DEAL_II
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
# 32 or more processes per node: UCX would stage GPU messages through the host
export UCX_NUM_PPN=4

mkdir -p $OUT
ldd $MG_BIN | grep -E 'libdeal_II|libkokkoscore'
ldd $HYBRID_BIN | grep -E 'libdeal_II|libkokkoscore'
echo "GPUs: $GPUS, sizes $MIN .. $MAX DoFs, runs: $RUNS, repetitions: $REPS"
echo "B: $N_CELLS cells; C: k=$K, $SWITCH_DOFS DoFs per rank; D: k=$K, $MATCH_SWITCH DoFs per rank"
echo "logs: $OUT"

common="--mpi=pspmix --nodes=$NODES --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
gpu_only="$common --ntasks=$GPUS --ntasks-per-node=4 --cpus-per-task=72"
hybrid="$common --ntasks=$((GPUS * K)) --ntasks-per-node=$((4 * K)) --cpus-per-task=$((72 / K))"

echo "placement of the hybrid:"
srun $hybrid bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V

for rep in $(seq 1 $REPS); do
  for run in $RUNS; do
    echo
    echo "=============== run $run repetition $rep"
    export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
    case $run in
      A) timeout $TIMEOUT srun $gpu_only $MG_BIN 4 $MIN $MAX 5 5 d 0 --single-rank-cells 0 ;;
      B) timeout $TIMEOUT srun $gpu_only $MG_BIN 4 $MIN $MAX 5 5 d 0 --single-rank-cells $N_CELLS ;;
      C) timeout $TIMEOUT srun $hybrid $HYBRID_BIN $K 4 $MIN $MAX 5 5 d $SWITCH_DOFS 0 ;;
      D) timeout $TIMEOUT srun $hybrid $HYBRID_BIN $K 4 $MIN $MAX 5 5 d $MATCH_SWITCH 0 ;;
      *) echo "unknown run $run" ;;
    esac | tee $OUT/${run}_r${rep}.log
  done
done

echo
echo "=============== cg_time [s], its ratio to A and the DoFs of the hybrid's switch level"
echo "GPUs: $GPUS, B: $N_CELLS cells, C: k=$K switch $SWITCH_DOFS, D: k=$K switch $MATCH_SWITCH"
python3 - $OUT $REPS $RUNS << 'EOF'
import os, sys

out, reps, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]


def last_table(name):
    """Rows of the last "n_proc cells dofs ..." table of a log, by fine-level DoFs."""
    with open(name) as f:
        lines = f.read().splitlines()
    starts = [i for i, line in enumerate(lines) if line.split()[:1] == ["n_proc"]]
    if not starts:
        return {}
    header = lines[starts[-1]].split()
    rows = {}
    for line in lines[starts[-1] + 1:]:
        values = line.split()
        if len(values) != len(header):
            break
        row = dict(zip(header, values))
        rows[int(row["dofs"])] = row
    return rows


for rep in range(1, reps + 1):
    runs = {}
    for run in names:
        name = os.path.join(out, "%s_r%d.log" % (run, rep))
        if os.path.isfile(name):
            runs[run] = last_table(name)
    if not runs.get("A"):
        print("repetition %d: no table of run A" % rep)
        continue

    print("repetition %d" % rep)
    print("%12s %4s" % ("dofs", "its") + "".join("%28s" % run for run in runs))
    for dofs, ref in sorted(runs["A"].items()):
        line = "%12d %4s" % (dofs, ref["cg_its"])
        for run, rows in runs.items():
            if dofs not in rows:
                line += "%28s" % "-"
                continue
            row = rows[dofs]
            time = float(row["cg_time"])
            text = "%.3e (%.2f)" % (time, time / float(ref["cg_time"]))
            if "switch_dofs" in row:
                text += " sw=%s" % row["switch_dofs"]
            if row["cg_its"] != ref["cg_its"]:
                text += " its=%s!" % row["cg_its"]
            line += "%28s" % text
        print(line)
EOF
