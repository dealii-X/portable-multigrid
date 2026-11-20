#!/bin/bash
#SBATCH --job-name=bk3-kernel-profile
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=01:00:00
#SBATCH --output=bk3-kernel-profile-%j.out

# Nsight Compute profile of the fine-level cell kernel of LaplaceOperatorBK3::vmult()
# in poisson_cube_bk3, for each BK3_VMULT_KERNEL (abstracted, plain, plain_constant)
# in double and in float: why is the abstracted kernel slower than the plain one in
# double but not in float? One program run per kernel and number type, a few launches
# of the fine-level kernel profiled with the full metric set.
#   KERNELS="abstracted plain" NUMBERS=double sbatch <this script>
# MIN_SIZE/MAX_SIZE bracket one cycle of the doubling mesh (default: 33.9M DoFs, Q4).
# The kernel is picked by its demangled name, see `regex` below; KERNEL_REGEX
# overrides it if the profiler prints the template arguments differently.
# Afterwards (also run at the end of the job):
#   python3 scripts/bk3_kernel_profile_summary.py <OUT>

set -u

# checkout and binary, override with REPO=... BIN=... sbatch
REPO=${REPO:-$HOME/codes/dealiiX-portable-multigrid}
BIN=${BIN:-$REPO/performance_tests/poisson_cube_multigrid/build/poisson_cube_bk3}

DEGREE=${DEGREE:-4}
KERNELS=${KERNELS:-"abstracted plain plain_constant"}
NUMBERS=${NUMBERS:-"double float"}
MIN_SIZE=${MIN_SIZE:-30000000}
MAX_SIZE=${MAX_SIZE:-40000000}
LAUNCH_SKIP=${LAUNCH_SKIP:-5}   # matching launches skipped before profiling
LAUNCH_COUNT=${LAUNCH_COUNT:-6} # matching launches profiled
OUT=${OUT:-$PWD/bk3-kernel-profile-${SLURM_JOB_ID:-local}}

source $REPO/scripts/jupiter/env.sh
module load Nsight-Compute 2>/dev/null || echo "no Nsight-Compute module, trying ncu from PATH"
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

if [ ! -x "$BIN" ]; then
  echo "no executable $BIN, build poisson_cube_bk3 first" >&2
  exit 1
fi

mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v ncu && ncu --version | tail -1
echo "degree $DEGREE, sizes $MIN_SIZE .. $MAX_SIZE DoFs, kernels: $KERNELS, numbers: $NUMBERS"
echo "launches: skip $LAUNCH_SKIP, profile $LAUNCH_COUNT; reports: $OUT"

NQ=$((DEGREE + 1))
opts="-n 1 --mpi=pspmix --cpu-bind=cores --gpu-bind=closest --gpus-per-task=1"

for number in $NUMBERS; do
  # fine-level instances: KokkosKernelAbstracted<dim, degree, n_q, Number> and
  # KokkosKernel<dim, n_dofs_1d, n_q, Number, ...>
  regex=${KERNEL_REGEX:-"KokkosKernelAbstracted<3, *$DEGREE, *$NQ, *$number|KokkosKernel<3, *$NQ, *$NQ, *$number"}
  for kernel in $KERNELS; do
    rep=$OUT/${kernel}_${number}
    echo
    echo "=============== kernel=$kernel number=$number regex=$regex"
    BK3_VMULT_KERNEL=$kernel srun $opts \
      ncu --set full --kernel-name-base demangled -k "regex:$regex" \
          --launch-skip $LAUNCH_SKIP --launch-count $LAUNCH_COUNT -o $rep --force-overwrite \
      $BIN $DEGREE $MIN_SIZE $MAX_SIZE 5 5 d 0
    if [ -f $rep.ncu-rep ]; then
      ncu --import $rep.ncu-rep --csv --page raw > $rep.csv
      echo "ncu csv: $rep.csv ($(wc -l < $rep.csv) lines)"
    else
      echo "no report $rep.ncu-rep: no launch matched the regex? set KERNEL_REGEX"
    fi
  done
done

echo
python3 $REPO/scripts/bk3_kernel_profile_summary.py $OUT
