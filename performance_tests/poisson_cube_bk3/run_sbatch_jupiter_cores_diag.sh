#!/bin/bash
#SBATCH --job-name=gpu-cores
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=gpu-cores-%j.out

# Why are the GPU levels of the hybrid program slower with many ranks per GPU?
# All levels on the GPUs, 1 node, CG solve [ms] (jobs 2861438, 2861440..46):
#   DoFs       bk3    hybrid k=18   k=36    k=72
#   1081665    50.4       54.2      58.6    69.8
#   2146689    60.5       65.5      70.8    83.6
#   4276737    66.7       71.6      77.1    90.4
#   8520321    76.6       79.8      84.7    93.9
#   16974593   86.8       91.1      96.1   107.2
# Two candidates: the GPU rank has only 72/k cores, or the other cores of the socket
# are busy (the CPU-only ranks spin while they wait).
# poisson_cube_bk3, 1 rank per GPU, Q4, 5 + 5 smoothing, the five sizes above:
#   A. the rank restricted to the first 72, 4, 2, 1 cores of its socket (as the GPU
#      rank at k = 1, 18, 36, 72), the other cores idle
#   B. 2 and 1 cores, with a busy loop on every other core of the socket
# Neither A nor B slower: the cause is specific to the waiting MPI ranks.

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $BK3_BIN | grep -E 'libdeal_II|libkokkoscore'

OUT=$PWD/gpu-cores-$SLURM_JOB_ID
mkdir -p $OUT
ARGS="4 1000000 17500000 5 5 d 0"

# wrap.sh <cores> <busy 0|1> <program and arguments>: runs the program on the first
# <cores> cores of the task's socket; busy = 1 adds a busy loop on each other core
cat > $OUT/wrap.sh <<'EOF'
#!/bin/bash
c=$1; busy=$2; shift 2
list=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
if [[ ! $list =~ ^[0-9]+-[0-9]+$ ]]; then echo "rank $SLURM_PROCID: unexpected core list $list"; exit 1; fi
first=${list%-*}; last=${list#*-}
pids=()
if [ "$busy" = 1 ]; then
  for ((i = first + c; i <= last; i++)); do
    taskset -c $i bash -c 'while :; do :; done' &
    pids+=($!)
  done
fi
echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES socket cores $list, program on $first-$((first + c - 1)), busy loops: ${#pids[@]}"
taskset -c $first-$((first + c - 1)) "$@"
rc=$?
if [ ${#pids[@]} -gt 0 ]; then kill "${pids[@]}"; fi
exit $rc
EOF
chmod +x $OUT/wrap.sh

# run <label> <cores> <busy>: one run, copied to $OUT/<label>.out
run() {
  local label=$1 c=$2 busy=$3
  echo
  echo "=============== $label: $c core(s) per rank, busy loops $busy: $BK3_BIN $ARGS"
  timeout 600 srun --mpi=pspmix --ntasks=4 --cpus-per-task=72 --distribution=block:block \
    --cpu-bind=cores --gpu-bind=closest --mem-bind=local \
    $OUT/wrap.sh $c $busy $BK3_BIN $ARGS 2>&1 | tee $OUT/$label.out
}

echo "######## A. cores of the GPU rank, other cores idle"
run c72 72 0
run c4 4 0
run c2 2 0
run c1 1 0

echo "######## B. other cores of the socket busy"
run c2-busy 2 1
run c1-busy 1 1

echo
echo "######## summary: final table of every run"
for l in c72 c4 c2 c1 c2-busy c1-busy; do
  echo "--- $l"
  awk '/cg_time/ {buf = ""} {buf = buf $0 "\n"} END {printf "%s", buf}' $OUT/$l.out | awk 'NF == 8 && $1 ~ /^(n_proc|[0-9]+)$/'
done
