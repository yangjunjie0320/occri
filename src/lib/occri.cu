#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#define THREAD_PER_BLOCK 256

__global__ void mo1T_mo2Tconj_to_rhoR(
    size_t ni, size_t ng,
    const cuDoubleComplex* x,
    const cuDoubleComplex* y,
    cuDoubleComplex* out
) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y;

    if (g >= ng || i >= ni) {
        return;
    }

    cuDoubleComplex xg = cuConj(x[g]);  // x should be conjugated (mo1T.conj())
    cuDoubleComplex yig = y[i * ng + g];
    out[i * ng + g] = cuCmul(xg, yig);
}

// will modify y in place
__global__ void coulg_rhoG_to_vG(
    size_t ni, size_t ng,
    const cuDoubleComplex* x,
    cuDoubleComplex* y
) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y;

    if (g >= ng || i >= ni) {
        return;
    }

    cuDoubleComplex xg = x[g];
    y[i * ng + g] = cuCmul(y[i * ng + g], xg);
}


__global__ void vR_mo2Tconj_to_out(
    size_t ni, size_t ng,
    const cuDoubleComplex* x,
    const cuDoubleComplex* y,
    cuDoubleComplex* out
) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (g >= ng) {
        return;
    }
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    
    for (size_t i = 0; i < ni; ++i) {
        cuDoubleComplex xig = x[i * ng + g];
        cuDoubleComplex yig = cuConj(y[i * ng + g]);
        sum = cuCadd(sum, cuCmul(xig, yig));
    }
    
    out[g] = sum;
}

extern "C" int OccRI_vR_dot_dm(
    int nmo1, int nmo2, 
    int ngrids, int* mesh, int blksize,
    const cuDoubleComplex* mo1T, 
    const cuDoubleComplex* mo2T, 
    const cuDoubleComplex* coulg, 
    cuDoubleComplex* out, 
    cuDoubleComplex* work
) {
    cufftHandle plan;
    cufftPlanMany(
        &plan, 3, mesh, 
        NULL, 1, ngrids, // Input layout
        NULL, 1, ngrids, // Output layout
        CUFFT_Z2Z, blksize
    );

    int blocks = (ngrids + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    dim3 td(THREAD_PER_BLOCK);

    for (int i = 0; i < nmo1; ++i) {
        const cuDoubleComplex* mo1T_i = mo1T + i * ngrids;

        for (int j0 = 0; j0 < nmo2; j0 += blksize) {
            int j1 = min(j0 + blksize, nmo2);
            const cuDoubleComplex* mo2T_j0_j1 = mo2T + j0 * ngrids;

            dim3 gd1(blocks, j1 - j0);
            dim3 gd2(blocks, 1);
            
            // "g,ig->ig", mo1T_i.conj(), mo2T_j0_j1
            cuDoubleComplex* rhoR = work; 
            mo1T_mo2Tconj_to_rhoR<<<gd1, td>>>(j1 - j0, ngrids, mo1T_i, mo2T_j0_j1, rhoR);
            
            // FFT (In-place)
            cuDoubleComplex* rhoG = work;
            cufftExecZ2Z(plan, rhoR, rhoG, CUFFT_FORWARD);
            
            // "g,ig->ig", coulg, vG
            cuDoubleComplex* vG = work;
            coulg_rhoG_to_vG<<<gd1, td>>>(j1 - j0, ngrids, coulg, vG);

            // Inverse FFT (In-place)
            cuDoubleComplex* vR = work;
            cufftExecZ2Z(plan, vG, vR, CUFFT_INVERSE);

            // "ig,ig->g", vR, mo2T_j0_j1.conj() - accumulate to out
            vR_mo2Tconj_to_out<<<gd2, td>>>(j1 - j0, ngrids, vR, mo2T_j0_j1, out + i * ngrids);
        }
    }

    cudaDeviceSynchronize();
    cufftDestroy(plan);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in OccRI_vR_dot_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
