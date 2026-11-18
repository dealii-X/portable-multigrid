#!/bin/bash
#SBATCH --job-name=gpu-busy-load
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --output=gpu-busy-load-%j.out

# Do busy Grace cores slow the GPU of the same module when the GPU is at full load?
# In the hybrid program the computation of one fine-level operator application at 135M
# DoFs per GPU takes 5.72 ms GPU-only, 5.76 ms with 4 ranks per GPU and 5.90 - 5.96 ms
# with 18 (jobs 2882551, 2882892, 2882546, 2884619); it grows with the number of ranks,
# appears on one GPU as well and is not changed by MPS. Every waiting rank keeps one
# core spinning. Job 2880860 tested busy cores only up to 4M DoFs per GPU.
# poisson_cube_bk3, 1 rank per GPU, Q4, 5 + 5 smoothing, 538970625 DoFs (135M per GPU):
#   ref      72 cores per rank, no other load
#   busy17   rank on 4 cores, busy loop on 17 other cores of its Grace (as k = 18)
#   busy35   rank on 2 cores, 35 busy cores (as k = 36)
#   busy71   rank on 1 core, 71 busy cores (as k = 72)
#   ref2     as ref, at the end (drift of the node during the job)
# Power draw, temperature and utilization of the GPUs are sampled every 2 s.
# Hybrid, all levels on the GPUs, for comparison: k = 18 computation 5.91 ms (+3%).

# cpu-tests checkout and build directory name, override with REPO=... BUILD=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-cpu-tests/tests}
BUILD=${BUILD:-build-neon}
BK3_BIN=$REPO/poisson_cube_bk3/$BUILD/program
DOFS=${DOFS:-538970625}

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd $BK3_BIN | grep -E 'libdeal_II|libkokkoscore'

OUT=$PWD/gpu-busy-load-$SLURM_JOB_ID
mkdir -p $OUT
ARGS="4 $((DOFS - 1000)) $((DOFS + 1000)) 5 5 d 0"

# wrap.sh <cores> <busy 0|1> <program and arguments>: runs the program on the first
# <cores> cores of the task's Grace; busy = 1 adds a busy loop on the first core of
# every further group of <cores> cores (the cores the waiting ranks of the hybrid
# program would spin on)
cat > $OUT/wrap.sh <<'EOF'
#!/bin/bash
c=$1; busy=$2; shift 2
list=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
if [[ ! $list =~ ^[0-9]+-[0-9]+$ ]]; then echo "rank $SLURM_PROCID: unexpected core list $list"; exit 1; fi
first=${list%-*}; last=${list#*-}
pids=()
if [ "$busy" = 1 ]; then
  for ((i = first + c; i <= last; i += c)); do
    taskset -c $i bash -c 'while :; do :; done' &
    pids+=($!)
  done
fi
echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES Grace cores $list, program on $first-$((first + c - 1)), busy loops: ${#pids[@]}"
taskset -c $first-$((first + c - 1)) "$@"
rc=$?
if [ ${#pids[@]} -gt 0 ]; then kill "${pids[@]}"; fi
exit $rc
EOF
chmod +x $OUT/wrap.sh

# run <label> <cores> <busy>: one run, copied to $OUT/<label>.out; GPU samples
# (index, power W, temperature C, utilization %) in $OUT/<label>.gpu
run() {
  local label=$1 c=$2 busy=$3
  echo
  echo "=============== $label: $c core(s) per rank, busy loops $busy: $BK3_BIN $ARGS"
  nvidia-smi --query-gpu=index,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 2 > $OUT/$label.gpu &
  local sampler=$!
  # throttle state, in a second sampler: if the field is not known here, only this one fails
  nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv,noheader -l 2 > $OUT/$label.throttle 2> $OUT/$label.throttle.err &
  local sampler2=$!
  local t0=$SECONDS
  timeout 900 srun --mpi=pspmix --ntasks=4 --cpus-per-task=72 --distribution=block:block \
    --cpu-bind=cores --gpu-bind=closest --mem-bind=local \
    $OUT/wrap.sh $c $busy $BK3_BIN $ARGS 2>&1 | tee $OUT/$label.out
  echo "srun elapsed: $((SECONDS - t0)) s"
  kill $sampler $sampler2 2>/dev/null
  wait $sampler $sampler2 2>/dev/null
}

run ref 72 0
run busy17 4 1
run busy35 2 1
run busy71 1 1
run ref2 72 0

echo
echo "######## summary"
for l in ref busy17 busy35 busy71 ref2; do
  [ -f $OUT/$l.out ] || continue
  echo "--- $l"
  grep -E "Best timings for ndof = $DOFS +mv " $OUT/$l.out
  grep -E "Best timings for ndof = $DOFS +on level [0-9]+\| +ghost" $OUT/$l.out |
    awk '{for (i = 1; i <= NF; i++) if ($i == "level") lev = $(i + 1) + 0; if (lev > max) max = lev; line[NR] = $0; level[NR] = lev}
         END {for (i = 1; i <= NR; i++) if (level[i] == max) print line[i]}'
  # GPUs under load (utilization >= 90%): mean and maximum power, maximum temperature
  awk -F', ' '$4 + 0 >= 90 {n[$1]++; p[$1] += $2; if ($2 + 0 > pm[$1]) pm[$1] = $2 + 0; if ($3 + 0 > tm[$1]) tm[$1] = $3 + 0}
              END {for (g = 0; g < 8; g++) if (g in n) printf "GPU %d under load: %d samples, power mean %.0f W, max %.0f W, max temperature %d C\n", g, n[g], p[g] / n[g], pm[g], tm[g]}' $OUT/$l.gpu
  if [ -s $OUT/$l.throttle ]; then
    echo "throttle reasons seen: $(awk -F', ' '{print $2}' $OUT/$l.throttle | sort | uniq -c | tr '\n' ';')"
  else
    echo "throttle reasons: not available ($(head -n 1 $OUT/$l.throttle.err))"
  fi
done
