from pyscf import lib
import cupy as cp
from gpu4pyscf.pbc.dft.multigrid_v2 import fft_in_place, ifft_in_place

def get_full_jks(s, v, c):
    sc = cp.dot(s, c)
    ccs = cp.dot(c, sc.T.conj())
    scv = cp.dot(sc, v)
    u = scv + scv.T.conj()
    u -= cp.dot(u, ccs)
    return u

def _version1(out, buffer, mo1T, mo2T, coulg, mesh=None):
    nmo1 = mo1T.shape[0]
    nmo2 = mo2T.shape[1]
    blksize = buffer.shape[0]
    ngrids = cp.prod(mesh)

    for i in range(0, nmo1, blksize):
        for j0, j1 in lib.prange(0, nmo2, blksize):
            # rhoR = buffer[:j1-j0]
            # rhoR += mo1T[i].conj() * mo2T[j0:j1]
            # rhoR = rhoR.reshape(-1, *mesh)
            # rhoR = buffer[:j1-j0]
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