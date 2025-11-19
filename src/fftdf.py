def FFTDF(cell=None, kpts=None, save_memory=True, with_occri=False, use_cpu=False, use_gpu=False, version=None):
    assert not (use_cpu and use_gpu)
    assert use_cpu or use_gpu

    df_obj = None
    if version == "pyscf":
        import pyscf, gpu4pyscf
        from pyscf.pbc.df.fftdf import FFTDF
        df_obj = FFTDF()
        if use_gpu:
            import gpu4pyscf.pbc.df.fftdf
            df_obj = df_obj.to_gpu()

        assert df_obj is not None
        return df_obj

    assert with_occri and use_gpu
    assert save_memory
    df_obj = OccRI(cell, kpts)
    return df_obj

import cupy as cp
from gpu4pyscf.pbc.df import fft

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.df_jk import _format_dms, _format_jks

import vR_dot_dm
from vR_dot_dm import get_full_jks

from cupyx.profiler import time_range
@time_range("get_k_kpts_occri")
def get_k_kpts_occri(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), 
                           kpts_band=None, exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension == 3
    coords = mydf.grids.coords
    ngrids = coords.shape[0]

    assert hermi == 1
    assert kpts_band is None
    input_band = None

    kpts = np.asarray(kpts)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = (cell.vol / ngrids) / nkpts 

    dtype = np.complex128
    if is_zero(kpts):
        dtype = dms.dtype
    assert dtype in [np.float64, np.complex128]
    vk_kpts = cp.zeros((nset, nkpts, nao, nao), dtype=dtype)

    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    mo_occ = getattr(dm_kpts, 'mo_occ', None)
    if mo_coeff is not None and mo_occ is not None:
        mo_coeff = cp.asarray(mo_coeff).reshape(nset, nkpts, nao, -1)
        mo_occ = cp.asarray(mo_occ).reshape(nset, nkpts, -1)
    else:
        return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

    blksize = mydf.blksize
    from gpu4pyscf.pbc.dft.numint import eval_ao
    for s in range(nset):
        cocc_kpts = []
        nocc_kpts = []
        for k in range(nkpts):
            mask = mo_occ[s, k] > 0
            cocc_kpts.append(mo_coeff[s, k][:, mask])
            nocc_kpts.append(mo_occ[s, k][mask])

        for k1 in range(nkpts):
            kpt1 = kpts[k1]
            ao1 = eval_ao(cell, coords, kpt=kpt1)

            cocc1 = cocc_kpts[k1]            
            mo1 = cp.dot(ao1, cocc1)
            nmo1 = mo1.shape[1]
            mo1T = mo1.T.reshape(nmo1, 1, ngrids)

            vR_dm = cp.zeros((nmo1, ngrids), dtype=dtype)
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

                cocc2 = cocc_kpts[k2]
                mo2 = cp.dot(ao2, cocc2 * nocc_kpts[k2] ** 0.5)
                nmo2 = mo2.shape[1]
                mo2T = mo2.T.reshape(1, nmo2, ngrids)

                for i0, i1 in lib.prange(0, nmo1, blksize):
                    rhoR = mo1T[i0:i1].conj() * mo2T
                    rhoR = rhoR.reshape(-1, ngrids)

                    rhoG = tools.fft(rhoR, mesh)
                    vG = rhoG * coulg
                    rhoR = rhoG = None

                    vR = tools.ifft(vG, mesh).reshape(i1 - i0, nmo2, ngrids)
                    vR_dm_i0i1 = contract('ijg,jg->ig', vR, mo2T[0].conj())
                    vR = vG = None

                    if dtype == np.float64:
                        vR_dm_i0i1 = vR_dm_i0i1.real
                    vR_dm[i0:i1] += vR_dm_i0i1
            
            ovlp1 = mydf.ovlp_kpts[k1]
            vk_k1 = cp.dot(vR_dm, ao1) * weight
            vk_kpts[s, k1] = get_full_jks(ovlp1, vk_k1, cocc1)

    if exxdiv == 'ewald':
        from gpu4pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

class OccRI(fft.FFTDF):
    blksize = 32

    def __init__(self, cell, kpts):
        super().__init__(cell, kpts)
        self.ovlp_kpts = None

    def build(self):
        self.dump_flags()

        from gpu4pyscf.pbc.gto import int1e
        cell = self.cell
        kpts = self.kpts
        self.ovlp_kpts = int1e.int1e_ovlp(cell, kpts)

    def dump_flags(self):
        from pyscf.lib import logger
        log = logger.new_logger(self, self.verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('\n')
        log.info('******** %s ********', self.__class__)
        
        mesh = self.mesh
        ngrids = cp.prod(mesh)
        log.info('mesh = %s (%d PWs)', mesh, ngrids)

        kpts = self.kpts
        nkpts = len(kpts)
        log.info('nkpts = %d', nkpts)

        cell = self.cell
        nao = cell.nao_nr()
        nocc = max(cell.nelec)
        blksize = min(self.blksize, nocc)
        log.info('nao = %d, nocc = %d, blksize = %d', nao, nocc, blksize)
        log.info('rhoR memory usage = %6.2e GB', blksize * nocc * ngrids * 16 / 1e9)
        log.info('ao_kpts memory usage = %6.2e GB', nkpts * nao * ngrids * 16 / 1e9)

        return self

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        assert omega is None
        if self.ovlp_kpts is None:
            self.build()

        from gpu4pyscf.pbc.df import fft_jk
        from gpu4pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(kpts, dm)
        if is_single_kpt:
            vj = vk = None
            kpt = kpts[0].reshape(1, 3)
            raise NotImplementedError #TODO

        else:
            vj = vk = None
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)

            if with_k:
                vk = get_k_kpts_occri(self, dm, hermi, kpts, kpts_band, exxdiv)

        return vj, vk

    # def get_pp(self, kpts=None):
    #     import get_pp
    #     return get_pp.get_pp(self, kpts)
        