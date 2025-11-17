#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=gpu4pyscf-occri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=04:00:00
#SBATCH --qos=debug

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source $HOME/anaconda3/bin/activate

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

ENV_NAME="gpu4pyscf-occri"
ENV_PATH=$(conda info --base)/envs/${ENV_NAME}

rm -rf $ENV_PATH

if ! [ -d ${ENV_PATH} ]; then
    conda create -n ${ENV_NAME} python=3.12 -y
    conda activate ${ENV_NAME}

    conda install -c conda-forge cmake compilers -y
    conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
    
    export CC=${ENV_PATH}/bin/x86_64-conda-linux-gnu-gcc
    export CXX=${ENV_PATH}/bin/x86_64-conda-linux-gnu-g++
    export FC=${ENV_PATH}/bin/x86_64-conda-linux-gnu-gfortran
    export CUDAHOSTCXX=${ENV_PATH}/bin/x86_64-conda-linux-gnu-g++

    export CUDA_HOME=${ENV_PATH}
    export CUDA_PATH=${ENV_PATH}
    export CUDAARCHS="90"
    export CMAKE_CONFIGURE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"
    
    pip install pyscf==2.8.0 geometric==1.1.0 basis-set-exchange==0.11
    pip install gpu4pyscf-libxc-cuda12x==0.5.0 pyscf-dispersion==1.3.0
    pip install --no-deps -v git+https://github.com/pyscf/gpu4pyscf.git@master
    
    pip install cupy-cuda12x==13.4.1 
    pip install cutensor-cu12==2.2.0 

    python -c "import cupy; import gpu4pyscf"
    conda deactivate
fi


conda activate ${ENV_NAME}
python ./work/nio-test/main.py 
