conda activate gpu4pyscf-occri

export CUDA_HOME=$CONDA_PREFIX

nvcc occri.cu -o liboccri.so -shared -std=c++17 \
    -arch=sm_90 \
    -I$CUDA_HOME/include -L$CUDA_HOME/lib64 \
    -lcufft -lcudart -O3 -Xcompiler -fPIC