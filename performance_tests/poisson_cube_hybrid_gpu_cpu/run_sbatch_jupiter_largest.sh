#!/bin/bash
#SBATCH --job-name=mg-largest
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:30:00
#SBATCH --output=mg-largest-%j.out

# What is the largest problem that fits on one node, and how do GPU-only, hybrid and
# CPU-only compare there? Largest size so far: 269748225 DoFs (67M per GPU, jobs
# 2864918, 2864928), CG solve 344.2 ms GPU-only, 324.2 ms hybrid k=4, 1105 ms CPU-only.
# Q4, 5 + 5 smoothing, float V-cycle. Every run is one size, with the GPU memory
# sampled every 5 s; the peak per GPU is printed after the run.
#   1. poisson_cube_bk3, 1 rank per GPU, on larger and larger meshes until a run
#      fails (out of memory). LADDER: DoFs:mesh, d = doubling mesh, s = cube with
#      224 cells per direction (a step between two doubling cycles).
#   2. hybrid, k = 18, switch_dofs = 8000, at 135M and 270M DoFs (not yet run with k=18)
#   3. for the doubling sizes that bk3 solved: hybrid, and poisson_cube_cpu on 288
#      ranks with block placement
# The summary at the end lists every run: solved or not, peak GPU memory, CG time.
# Quick variant, bk3 on one GPU only, ladder in DoFs on that GPU:
#   GPUS=1 LADDER_ONLY=1 LADDER="135005697:d 269748225:d 454756609:s 538970625:d 721734273:s 1076890625:d" sbatch <this script>

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
CPU_BIN=$REPO/poisson_cube_cpu/$BUILD/program
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program
HYB_BIN=$REPO/poisson_cube_hybrid_gpu_cpu/$BUILD/program
K=${K:-18}
GPUS=${GPUS:-4}
LADDER_ONLY=${LADDER_ONLY:-0}

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $HYB_BIN | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g

OUT=$PWD/mg-largest-$SLURM_JOB_ID
mkdir -p $OUT

GPU_OPTS="--mpi=pspmix --ntasks=$GPUS --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
HYB_OPTS="--mpi=pspmix --ntasks=$((4 * K)) --cpus-per-task=$((72 / K)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
CPU_OPTS="--mpi=pspmix --ntasks=288 --cpus-per-task=1 --distribution=block:block --cpu-bind=cores --mem-bind=local"

# run <label> <srun options> <program and arguments>: one run, copied to
# $OUT/<label>.out, GPU memory samples in $OUT/<label>.mem
run() {
  local label=$1 opts=$2; shift 2
  echo
  echo "=============== $label: srun $opts $*"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 5 > $OUT/$label.mem &
  local sampler=$!
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  local t0=$SECONDS
  timeout 5400 srun $opts "$@" 2>&1 | tee $OUT/$label.out
  echo "srun elapsed: $((SECONDS - t0)) s"
  kill $sampler
  wait $sampler 2>/dev/null
  awk -F', ' '{if ($2 + 0 > m[$1]) m[$1] = $2 + 0; t[$1] = $3 + 0}
              END {for (g in m) printf "GPU %s: peak %d MiB of %d MiB\n", g, m[g], t[g]}' $OUT/$label.mem | sort
}

# size <DoFs>: the program arguments min_size max_size that select this one cycle
size() { echo "$(($1 - 1000)) $(($1 + 1000))"; }

# solved <label> <DoFs>: did the run print a CG time for this size?
solved() { grep -q -E "^ *([0-9]+ +)?[0-9]+ +$2 +[0-9.e+-]+ +[0-9.e+-]+ +[0-9.e+-]+ +[0-9]+ " $OUT/$1.out; }

LADDER=${LADDER:-"538970625:d 721734273:s 1076890625:d 2152730625:d 3630961153:s"}

echo "######## 1. bk3 on $GPUS GPU(s) until a run fails"
FIT=""
for rung in $LADDER; do
  dofs=${rung%:*}
  mesh=${rung#*:}
  run bk3-$dofs "$GPU_OPTS" $BK3_BIN 4 $(size $dofs) 5 5 $mesh 0
  if ! solved bk3-$dofs $dofs; then
    echo "######## bk3 did not solve $dofs DoFs ($((dofs / GPUS)) per GPU), stopping the ladder"
    break
  fi
  if [ $mesh = d ]; then FIT="$FIT $dofs"; fi
done

if [ "$LADDER_ONLY" != 1 ]; then

hybrid() { UCX_NUM_PPN=4 run hybrid-$1 "$HYB_OPTS" $HYB_BIN $K 4 $(size $1) 5 5 d 8000 0; }

echo "######## 2. hybrid k=$K, sizes that are known to fit"
hybrid 135005697
hybrid 269748225

echo "######## 3. hybrid k=$K and CPU-only at the doubling sizes bk3 solved:$FIT"
for dofs in $FIT; do
  hybrid $dofs
  run cpu-$dofs "$CPU_OPTS" $CPU_BIN 4 $(size $dofs) 5 5 d 0
done

fi

echo
echo "######## summary: run, solved or not, peak GPU memory [MiB], row of the final table"
for f in $(ls $OUT/*.mem | sort -V); do
  label=$(basename $f .mem)
  dofs=${label##*-}
  if solved $label $dofs; then state=solved; else state="NOT SOLVED"; fi
  echo "--- $label: $state; $(awk -F', ' '{if ($2 + 0 > m[$1]) m[$1] = $2 + 0} END {for (g = 0; g < 8; g++) if (g in m) printf "GPU %d %d  ", g, m[g]}' $f)"
  grep -E "^ *([0-9]+ +)?[0-9]+ +$dofs +[0-9.e+-]+ +[0-9.e+-]+ +[0-9.e+-]+ +[0-9]+ " $OUT/$label.out | tail -n 1
  if [ "$state" != solved ]; then grep -m 3 -i -E "failed to allocate|out of memory|bad_alloc|oom|Kokkos ERROR|Killed" $OUT/$label.out; fi
done
