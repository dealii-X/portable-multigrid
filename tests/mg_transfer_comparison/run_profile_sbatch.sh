#!/bin/bash
#SBATCH --job-name=mg-transfer-profile
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=00:45:00
#SBATCH --output=results/profile-%j.out
#
# Profile the transfer kernels of one mesh size with Nsight Systems (kernel
# time summary) and Nsight Compute (full metric set on every transfer kernel
# launch), to locate where the copy's restriction loses against deal.II and
# BK1 at high degree.
#
# Submit from this directory with the benchmark built (see run_sbatch.sh):
#
#   sbatch run_profile_sbatch.sh
#   DEGREE=4 MIN_SIZE=30000000 MAX_SIZE=70000000 sbatch run_profile_sbatch.sh
#
# MIN_SIZE/MAX_SIZE should bracket a single cycle of the doubling mesh; the
# defaults select the ~1.3e8 fine dof cycle of degree 5. N_APPLICATIONS
# (default 2) applications per timing repetition keep the number of profiled
# launches small: with 5 repetitions plus the correctness application, each
# of the 4 transfer variants launches 11 prolongation and 11 restriction
# kernels per cycle.
#
# Afterwards, from this directory:
#
#   python3 profile_summary.py results/profile-<jobid>-ncu.csv
#
# prints the key metrics of the deal.II, copy and BK1 kernels side by side.

set -u

DIM=${DIM:-3}
DEGREE=${DEGREE:-5}
MIN_SIZE=${MIN_SIZE:-100000000}
MAX_SIZE=${MAX_SIZE:-200000000}
MESH=${MESH:-d}
CELLS_PER_BATCH=${CELLS_PER_BATCH:-0}
N_APPLICATIONS=${N_APPLICATIONS:-2}
BUILD_DIR=${BUILD_DIR:-build}

source ../../scripts/jupiter/env.sh ${DEAL_II_DIR:-}

module load Nsight-Systems 2>/dev/null || module load Nsight-Systems/2025.5.1 2>/dev/null || echo "no Nsight-Systems module, trying nsys from PATH"
module load Nsight-Compute 2>/dev/null || echo "no Nsight-Compute module, trying ncu from PATH"

export KOKKOS_DEVICE_BACKEND=CUDA
export KOKKOS_DISABLE_OPENMP=true
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export CUDA_DEVICE_WAITS_ON_EXCEPTION=0

if [ ! -x "$BUILD_DIR/program" ]; then
  echo "no executable $BUILD_DIR/program, build first (see run_sbatch.sh)" >&2
  exit 1
fi

args=("$DIM" "$DEGREE" "$MIN_SIZE" "$MAX_SIZE" "$MESH" "$CELLS_PER_BATCH" "$N_APPLICATIONS")
out=results/profile-$SLURM_JOB_ID

echo "degree=$DEGREE dim=$DIM sizes=[$MIN_SIZE,$MAX_SIZE] mesh=$MESH cells_per_batch=$CELLS_PER_BATCH n_applications=$N_APPLICATIONS"
echo "executable: $BUILD_DIR/program"; echo "deal.II:    $DEAL_II_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v nsys && nsys --version | head -1
command -v ncu && ncu --version | tail -1
echo

# 1. Nsight Systems: time per kernel, all variants. Kernel names carry the
#    namespace (dealii::Portable vs dealii::Copy::Portable vs BK1).
echo "=== nsys ==="
srun -n 1 --mpi=pspmix --cpu-bind=cores --gpu-bind=closest --gpus-per-task=1 \
  nsys profile -o "$out-nsys" -t cuda,nvtx --stats=true --force-overwrite=true \
  "$BUILD_DIR/program" "${args[@]}"
echo

# 2. Nsight Compute: full metric set on every launch of the transfer kernels
#    (the launch-count bounds the time: ~60 launches x full replay).
echo "=== ncu ==="
srun -n 1 --mpi=pspmix --cpu-bind=cores --gpu-bind=closest --gpus-per-task=1 \
  ncu --set full --kernel-name-base demangled \
      -k 'regex:restrict_and_add_internal|prolongate_and_add_internal|KokkosRestriction|KokkosProlongation' \
      --launch-count 100 -o "$out-ncu" --force-overwrite \
  "$BUILD_DIR/program" "${args[@]}"
echo

# export the raw metrics of every profiled launch as CSV for profile_summary.py
ncu --import "$out-ncu.ncu-rep" --csv --page raw > "$out-ncu.csv"
echo "ncu csv: $out-ncu.csv ($(wc -l < "$out-ncu.csv") lines)"
python3 profile_summary.py "$out-ncu.csv"
