#!/bin/bash
#SBATCH --reservation=changroup-h100-node-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpu4pyscf-occri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=10:00:00

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source $HOME/anaconda3/bin/activate gpu4pyscf-occri

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=4;
export OPENBLAS_NUM_THREADS=4;
export PYSCF_MAX_MEMORY=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))

echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/resnick/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
ln -s $PYSCF_TMPDIR tmp

python main.py
# kernprof -l -v main.py

echo "End time = $(date)"
