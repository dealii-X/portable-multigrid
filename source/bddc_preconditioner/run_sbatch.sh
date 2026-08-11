#!/bin/bash
#SBATCH --job-name=dd-bnn
#SBATCH --partition=booster
#SBATCH --account=dealii-X
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1       
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=output-%j.out

module purge

module load Stages/2026

module load nvidia-compilers

module load ParaStationMPI

module load FlexiBLAS

export LD_LIBRARY_PATH=$WORK/sw/lib:$WORK/sw/lib64:$WORK/sw/dealii/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1

export KOKKOS_DEVICE_BACKEND=CUDA
export KOKKOS_DISABLE_OPENMP=true

export OMP_PROC_BIND=close
export OMP_PLACES=cores

export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export CUDA_DEVICE_WAITS_ON_EXCEPTION=0


srun --cpu-bind=cores \
     --gpu-bind=closest \
      --mem-bind=local \
      --gpus-per-task=1 \
     ./program

rm core.jpbo*.jupiter.*


                                        
