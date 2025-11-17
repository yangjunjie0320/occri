import cupy as cp

device = cp.cuda.Device()
print(f"Using GPU device {device.id}")

mem_info = cp.cuda.runtime.memGetInfo()
print(f"Free memory: {mem_info[0] / (1024**3):.2f} GB")
print(f"Total memory: {mem_info[1] / (1024**3):.2f} GB")

import pyscf, gpu4pyscf
from pyscf.pbc import gto
import gpu4pyscf.pbc
from gpu4pyscf.pbc.df import fft_jk

pcell = gto.Cell()
pcell.atom = """
Ni 0.0000 0.0000 0.0000
Ni 4.1700 4.1700 4.1700
O  2.0850 2.0850 2.0850
O  6.2550 6.2550 6.2550
"""
pcell.a = """
4.1700 2.0850 2.0850
2.0850 4.1700 2.0850
2.0850 2.0850 4.1700
"""
pcell.unit = "A"
pcell.basis = 'gth-dzvp-molopt-sr'
pcell.pseudo = "gth-hf-rev"
pcell.ke_cutoff = 200
pcell.exp_to_discard = 0.1
pcell.verbose = 5
pcell.build()

kpts = pcell.make_kpts([2, 2, 2])

from gpu4pyscf.pbc.scf import KRHF, KUHF
# mf = KRHF(pcell, kpts)
mf = KUHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')

mf.kernel(dm0)
ene_ref = mf.e_tot

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.df_jk import _format_dms, _format_jks

mf.with_df.ovlp_kpts = mf.get_ovlp()
def get_k_kpts_occri(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), 
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

    if mo_coeff is not None and mo_occ is not None:
        mo_coeff = cp.asarray(mo_coeff).reshape(nset, nkpts, nao, -1)
        mo_occ = cp.asarray(mo_occ).reshape(nset, nkpts, -1)
    else:
        return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

    blksize = 32

    from gpu4pyscf.pbc.dft.numint import eval_ao
    for s in range(nset):
        for k1 in range(nband):
            kpt1 = kpts_band[k1]
            ao1 = eval_ao(cell, coords, kpt=kpt1)

            mask = mo_occ[s, k1] > 0
            cocc1 = mo_coeff[s, k1][:, mask]
            
            mo1 = cp.dot(ao1, cocc1)
            mo1T = mo1.T.reshape(-1, 1, ngrids)
            nmo1 = mo1T.shape[0]

            vr_dm = cp.zeros((nmo1, ngrids), dtype=dtype)
            for k2 in range(nkpts):
                kpt2 = kpts[k2]
                ao2 = eval_ao(cell, coords, kpt=kpt2)

                k21 = kpt2 - kpt1
                coulg = tools.get_coulG(cell, k21, False, mydf, mesh)

                if not is_zero(k21):
                    k21 = cp.asarray(k21)
                    theta = cp.dot(coords, k21)
                    phase = cp.exp(-1j * theta)
                    ao2 = ao2 * phase.reshape(-1, 1)

                mask = mo_occ[s, k2] > 0
                mo2 = cp.dot(ao2, mo_coeff[s, k2][:, mask] * mo_occ[s, k2][mask] ** 0.5)
                mo2T = mo2.T.reshape(1, -1, ngrids)

                for i0, i1 in lib.prange(0, nmo1, blksize):
                    rhoR = mo1T[i0:i1].conj()* mo2T
                    rhoR = rhoR.reshape(-1, ngrids)

                    rhoG = tools.fft(rhoR, mesh)
                    vG = rhoG * coulg
                    rhoR = rhoG = None

                    vR = tools.ifft(vG, mesh).reshape(i1-i0, -1, ngrids)
                    vr_dm[i0:i1] += contract('ijg,gj->ig', vR, mo2.conj())
                    vR = vG = None

            ovlp1 = mydf.ovlp_kpts[k1]
            cinv1 = cp.dot(ovlp1, cocc1)
            ccinv = cp.dot(cocc1, cinv1.T.conj())
            
            # cinv_vr_dm = cp.dot(cinv1, vr_dm)
            # v1 = cp.dot(cinv_vr_dm, ao1) * weight
            v1 = cp.dot(vr_dm, ao1) * weight
            cinv_v1 = cp.dot(cinv1, v1)

            vk_kpts[s, k1] += cinv_v1 + cinv_v1.T.conj()
            vk_kpts[s, k1] -= cp.dot(cinv_v1, ccinv)

    if exxdiv == 'ewald':
        from gpu4pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

fft_jk.get_k_kpts = get_k_kpts_occri
mf.kernel(dm0)
ene_sol = mf.e_tot

print("ene_ref = %12.6f" % ene_ref)
print("ene_sol = %12.6f" % ene_sol)
print("error   = %6.2e" % abs(ene_ref - ene_sol))
