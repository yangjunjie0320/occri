from pyscf import lib
import numpy as np
import cupy as cp

from gpu4pyscf.pbc import tools
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft.multigrid_v2 import fft_in_place, ifft_in_place

def get_full_jks(s, v, c):
    sc = cp.dot(s, c)
    ccs = cp.dot(c, sc.T.conj())
    scv = cp.dot(sc, v)
    u = scv + scv.T.conj()
    u -= cp.dot(scv, ccs)
    return u

from cupyx.profiler import time_range

@time_range("vR_dot_dm_version1")
def _version1(mo1T, mo2T, coulg, blksize=16, mesh=None):
    nmo1 = mo1T.shape[0]
    nmo2 = mo2T.shape[0]
    ngrids = cp.prod(mesh)

    mo1T = mo1T.reshape(nmo1, 1, ngrids)
    mo2T = mo2T.reshape(1, nmo2, ngrids)

    out = cp.zeros((nmo1, ngrids), dtype=np.complex128)
    for i0, i1 in lib.prange(0, nmo1, blksize):
        rhoR = mo1T[i0:i1].conj() * mo2T
        rhoR = rhoR.reshape(-1, *mesh)

        rhoG = tools.fft(rhoR, mesh)
        vG = rhoG * coulg
        rhoR = rhoG = None

        vR = tools.ifft(vG, mesh).reshape(i1 - i0, nmo2, ngrids)
        out[i0:i1] += contract('ijg,jg->ig', vR, mo2T[0].conj())
        vR = vG = None

    return out

@time_range("vR_dot_dm_version2")
def _version2(mo1T, mo2T, coulg, blksize=16, mesh=None):
    nmo1 = mo1T.shape[0]
    nmo2 = mo2T.shape[0]
    ngrids = cp.prod(mesh)

    out = cp.zeros((nmo1, ngrids), dtype=mo1T.dtype)
    for i in range(nmo1):
        for j0, j1 in lib.prange(0, nmo2, blksize):
            rhoR = mo1T[i].conj() * mo2T[j0:j1]
            rhoR = rhoR.reshape(-1, *mesh)

            fft_in_place(rhoR)
            rhoG = rhoR.reshape(-1, ngrids)
            
            vG = rhoG * coulg
            vG = vG.reshape(-1, *mesh)
            ifft_in_place(vG)

            vR = vG.reshape(j1 - j0, ngrids)
            vR *= mo2T[j0:j1].conj()
            out[i] += vR.sum(axis=0)
            vG = vR = None

    return out

def load_library(libname):
    try:
        import os
        libpath = "/home/junjiey/work/occri/occri-main/src/lib"
        # libpath = libpath / "lib"
        return np.ctypeslib.load_library(libname, libpath)
    except OSError:
        raise

def _version3(mo1T, mo2T, coulg, blksize=16, mesh=None):
    import ctypes    
    nmo1 = mo1T.shape[0]
    nmo2 = mo2T.shape[0]

    ngrids = int(cp.prod(mesh))
    factor = 1.0 / ngrids
    coulg = coulg * factor

    assert coulg.dtype == cp.complex128
    
    out = cp.zeros((nmo1, ngrids), dtype=cp.complex128)
    work = cp.zeros((blksize, ngrids), dtype=cp.complex128)
    
    # 加载库
    liboccri = load_library('liboccri')
    
    # 调用 CUDA 函数
    err = liboccri.OccRI_vR_dot_dm(
        ctypes.c_int(nmo1),
        ctypes.c_int(nmo2),
        ctypes.c_int(ngrids),
        ctypes.cast(mesh.ctypes.data, ctypes.c_void_p),
        ctypes.c_int(blksize),
        ctypes.cast(mo1T.data.ptr, ctypes.c_void_p),
        ctypes.cast(mo2T.data.ptr, ctypes.c_void_p),
        ctypes.cast(coulg.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(work.data.ptr, ctypes.c_void_p),
    )
    
    if err != 0:
        raise RuntimeError(f'CUDA kernel OccRI_vR_dot_dm failed with error code {err}')
    
    return out

if __name__ == "__main__":
    blksize = 4
    nmo1 = 10
    nmo2 = 10
    
    mesh = np.array([11, 11, 11], dtype=np.int32)
    ng = np.prod(mesh)
    
    mo1T = cp.random.randn(nmo1 * ng) + 1j * cp.random.randn(nmo1 * ng)
    mo1T = mo1T.reshape(nmo1, ng)
    mo2T = cp.random.randn(nmo2 * ng) + 1j * cp.random.randn(nmo2 * ng)
    mo2T = mo2T.reshape(nmo2, ng)
    coulg = cp.random.randn(ng) + 1j * cp.random.randn(ng)

    out1 = _version1(mo1T, mo2T, coulg, blksize, mesh)
    out2 = _version2(mo1T, mo2T, coulg, blksize, mesh)
    out3 = _version3(mo1T, mo2T, coulg, blksize, mesh)

    err12 = cp.abs(out1 - out2).max()
    err13 = cp.abs(out1 - out3).max()
    err23 = cp.abs(out2 - out3).max()

    print("err12 = %6.2e" % err12)
    print("err13 = %6.2e" % err13)
    print("err23 = %6.2e" % err23)
