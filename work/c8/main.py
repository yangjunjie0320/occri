import cupy as cp

device = cp.cuda.Device()
print(f"Using GPU device {device.id}")
mem_info = cp.cuda.runtime.memGetInfo()
print(f"Free memory: {mem_info[0] / (1024**3):.2f} GB")
print(f"Total memory: {mem_info[1] / (1024**3):.2f} GB")

import pyscf, gpu4pyscf
import gpu4pyscf.pbc

from pyscf.pbc import gto
pcell = gto.Cell()
pcell.a = '''
3.5668  0.0000  0.0000
0.0000  3.5668  0.0000
0.0000  0.0000  3.5668
'''
pcell.atom = '''
C     0.0000  0.0000  0.0000    
C     0.8917  0.8917  0.8917
C     1.7834  1.7834  0.0000    
C     2.6751  2.6751  0.8917
C     1.7834  0.0000  1.7834
C     2.6751  0.8917  2.6751
C     0.0000  1.7834  1.7834
C     0.8917  2.6751  2.6751
'''
pcell.basis = 'gth-dzvp'
pcell.pseudo = 'gth-hf-rev'
pcell.verbose = 4
pcell.ke_cutoff = 100
pcell.build()

kpts = pcell.make_kpts([2, 2, 2])

from gpu4pyscf.pbc.scf import KRHF
mf = KRHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')

# mf.kernel(dm0)
# ene_ref = mf.e_tot

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools

from gpu4pyscf.pbc.df.df_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils

# import line_profiler
# @line_profiler.profile
def get_k_kpts_v1(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                  exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    coords = mydf.grids.coords
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    ni = mydf._numint
    kpts = np.asarray(kpts)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = (cell.vol / ngrids) / nkpts 

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    dtype = np.complex128
    if is_zero(kpts_band) and is_zero(kpts):
        dtype = dms.dtype
    vk_kpts = cp.zeros((nset, nband, nao, nao), dtype=dtype)

    # calculate the size of ao1_kpts
    size = nkpts * nao * ngrids * 16 / 1e9
    print("nkpts = %d, nao = %d, ngrids = %6.2e" % (nkpts, nao, ngrids))
    print("Size of ao1_kpts: %6.2e GB" % size)
    
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
    mem = get_avail_mem() / 1e9
    print("Available memory: %6.2e GB" % mem)

    ao2_kpts = ni.eval_ao(cell, coords, kpts=kpts)
    ao1_kpts = ao2_kpts if input_band is None else ni.eval_ao(cell, coords, kpts=kpts_band)

    if mo_coeff is not None and nset == 1:
        mo2_kpts = [
            ao.dot(mo[:,occ>0] * occ[occ>0]**.5)
            for occ, mo, ao in zip(mo_occ, mo_coeff, ao2_kpts)]
        ao2_kpts = mo2_kpts
    else:
        mo2_kpts = None

    blksize = 32
    vr_dm = cp.zeros((nao, ngrids), dtype=dtype)
    for s in range(nset):
        for k1, ao1 in enumerate(ao1_kpts):
            ao1T = ao1.T # shape (ngrids, naoi)
            kpt1 = kpts_band[k1]
            vr_dm *= 0.0
            
            for k2, ao2 in enumerate(ao2_kpts):
                ao2T = ao2.T
                kpt2 = kpts[k2]

                k21 = kpt2 - kpt1
                coulg = tools.get_coulG(cell, k21, False, mydf, mesh)

                if not is_zero(k21):
                    k21 = cp.asarray(k21)
                    theta = cp.dot(coords, k21)
                    phase = cp.exp(-1j * theta)
                    ao2T = ao2T * phase.reshape(1, -1)

                ao_dm = cp.dot(dms[s, k2], ao2T.conj()) if mo2_kpts is None else ao2T.conj()

                for i0, i1 in lib.prange(0, nao, blksize):
                    rhoR = contract('ig,jg->ijg', ao1T[i0:i1].conj(), ao2T)
                    rhoR = rhoR.reshape(-1, ngrids)

                    rhoG = tools.fft(rhoR, mesh)
                    vG = rhoG * coulg
                    # rhoR = rhoG = None

                    vR = tools.ifft(vG, mesh).reshape(i1-i0, -1, ngrids)
                    vr_dm[i0:i1] += contract('ijg,jg->ig', vR, ao_dm)
                    # vR = vG = None

            vk_kpts[s, k1] += weight * cp.dot(vr_dm, ao1)

    if exxdiv == 'ewald':
        from gpu4pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

from gpu4pyscf.pbc.df import fft_jk
fft_jk.get_k_kpts = get_k_kpts_v1
mf.kernel(dm0)
ene_ref = mf.e_tot

# print("ene_ref = %12.6f" % ene_ref)
# print("ene_sol = %12.6f" % ene_sol)
# print("error   = %6.2e" % abs(ene_ref - ene_sol))

# @line_profiler.profile
def get_k_kpts_v2(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), 
                        kpts_band=None, exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    coords = mydf.grids.coords
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    ni = mydf._numint
    kpts = np.asarray(kpts)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = (cell.vol / ngrids) / nkpts 

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    dtype = np.complex128
    if is_zero(kpts_band) and is_zero(kpts):
        dtype = dms.dtype
    vk_kpts = cp.zeros((nset, nband, nao, nao), dtype=dtype)

    # calculate the size of ao1_kpts
    size = nkpts * nao * ngrids * 16 / 1e9
    print("nkpts = %d, nao = %d, ngrids = %6.2e" % (nkpts, nao, ngrids))
    print("Size of ao1_kpts: %6.2e GB" % size)
    
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
    mem = get_avail_mem() / 1e9
    print("Available memory: %6.2e GB" % mem)

    ao2_kpts = ni.eval_ao(cell, coords, kpts=kpts)
    ao1_kpts = ao2_kpts if input_band is None else ni.eval_ao(cell, coords, kpts=kpts_band)

    if mo_coeff is not None and mo_occ is not None:
        mo_coeff = cp.asarray(mo_coeff).reshape(nset, nkpts, nao, -1)
        mo_occ = cp.asarray(mo_occ).reshape(nset, nkpts, -1)

    blksize = 32
    vr_dm = cp.zeros((nao, ngrids), dtype=dtype)

    for s in range(nset):
        for k1, ao1 in enumerate(ao1_kpts):
            ao1T = ao1.T # shape (ngrids, naoi)
            kpt1 = kpts_band[k1]
            vr_dm *= 0.0
            
            for k2, ao2 in enumerate(ao2_kpts):
                ao2T = ao2.T
                kpt2 = kpts[k2]

                k21 = kpt2 - kpt1
                coulg = tools.get_coulG(cell, k21, False, mydf, mesh)

                if not is_zero(k21):
                    k21 = cp.asarray(k21)
                    theta = cp.dot(coords, k21)
                    phase = cp.exp(-1j * theta)
                    ao2T = ao2T * phase.reshape(1, -1)

                # ao_dm = cp.dot(dms[s, k2], ao2T.conj()) if mo2_kpts is None else ao2T.conj()
                if mo_coeff is not None and mo_occ is not None:
                    mask = mo_occ[s, k2] > 0
                    ao_dm = cp.dot(ao2.conj(), mo_occ[s, k2][mask] ** 0.5 * mo_coeff[s, k2, :, mask])
                    ao2T = ao_dm.conj().T
                else:
                    ao_dm = cp.dot(ao2.conj(), dms[s, k2])

                for i0, i1 in lib.prange(0, nao, blksize):
                    rhoR = ao1T[i0:i1].conj().reshape(-1, 1, ngrids) * ao2T.reshape(1, -1, ngrids)
                    rhoR = rhoR.reshape(-1, ngrids)

                    rhoG = tools.fft(rhoR, mesh)
                    vG = rhoG * coulg
                    rhoR = rhoG = None

                    vR = tools.ifft(vG, mesh).reshape(i1-i0, -1, ngrids)
                    vr_dm[i0:i1] += contract('ijg,jg->ig', vR, ao_dm.T)
                    vR = vG = None

            vk_kpts[s, k1] += weight * cp.dot(vr_dm, ao1)

    if exxdiv == 'ewald':
        from gpu4pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


fft_jk.get_k_kpts = get_k_kpts_v2
mf.kernel(dm0)
ene_sol = mf.e_tot

print("ene_ref = %12.6f" % ene_ref)
print("ene_sol = %12.6f" % ene_sol)
print("error   = %6.2e" % abs(ene_ref - ene_sol))