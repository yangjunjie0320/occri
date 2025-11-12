echo "Start time = $(date)"
export CUDA_VISIBLE_DEVICES=0;
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source $HOME/miniconda3/bin/activate gpu4pyscf-occri

export OMP_NUM_THREADS=32;
export MKL_NUM_THREADS=4;
export OPENBLAS_NUM_THREADS=4;
export PYSCF_MAX_MEMORY=32000; # 32GB

echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/home/yangjunjie/work/tmp/
export TMPDIR=$TMP/occri/
export PYSCF_TMPDIR=$TMPDIR
echo TMPDIR = $TMPDIR

python main-2.py
# compute-sanitizer --tool memcheck python main.py
# echo "End time = $(date)"
