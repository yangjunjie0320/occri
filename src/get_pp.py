import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import lib
from pyscf.pbc.df import fft as fft_cpu
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.gto import pseudo
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df import fft_jk
from gpu4pyscf.pbc.df.aft import _check_kpts
from gpu4pyscf.pbc.df.ft_ao import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import reset_kpts

from gpu4pyscf.pbc.df.fft import get_SI

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    from gpu4pyscf.pbc.dft import numint
    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    cell = mydf.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    mesh = mydf.mesh

    Gv = cell.get_Gv(mesh)
    Rv = cell.get_uniform_grids(mesh)
    SI = get_SI(cell, mesh=mesh)

    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -cp.einsum('ij,ij->j', SI, cp.asarray(vpplocG))
    vpplocR = tools.ifft(vpplocG, mesh).real

    ngrids = len(vpplocG)
    nkpts = len(kpts)
    nao = cell.nao
    if is_zero(kpts):
        vpp = cp.zeros((nkpts,nao,nao))
    else:
        vpp = cp.zeros((nkpts,nao,nao), dtype=np.complex128)

    kpts = np.asarray(kpts)
    for k, kpt in enumerate(kpts):
        ao = numint.eval_ao(cell, mydf.grids.coords, kpt=kpt)
        vpp[k] += (ao.conj().T * vpplocR).dot(ao)
        ao = None
    vpplocR = None

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    from gpu4pyscf.lib.cupy_helper import get_avail_mem, print_mem_info
    print_mem_info()

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = np.empty((48, ngrids), dtype=np.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)

        nao = cell.nao
        aokG = cp.empty((ngrids, nao), dtype=np.complex128)
        for g0, g1 in lib.prange(0, ngrids, 20000):
            gk_g0g1 = Gk[g0:g1]
            aokG[g0:g1] = ft_ao(cell, gk_g0g1, kpt=kpt) 
            print("kpt = %s, g0 = %d, g1 = %d" % (kpt, g0, g1))
        aokG *= (1 / cell.vol)**.5

        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = cp.asarray(buf[:p1])
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = SPG_lmi.dot(aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = cp.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = contract('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += contract('imp,imq->pq', SPG_lm_aoG.conj(), tmp)

        aokG = None
        return vppnl * (1./cell.vol)

    for k, kpt in enumerate(kpts):
        print("k = %d, kpt = %s" % (k, kpt))
        vppnl = vppnl_by_k(kpt)
        if is_zero(kpt):
            vpp[k] += cp.asarray(vppnl.real)
        else:
            vpp[k] += cp.asarray(vppnl)

    if is_single_kpt:
        vpp = vpp[0]
    return vpp