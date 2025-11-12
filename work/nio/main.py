import cupy as cp, numpy as np
import pyscf, gpu4pyscf
import gpu4pyscf.pbc

def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, cutoff=None, out=None,
                 verbose=None, opt=None):

    from gpu4pyscf.lib import logger
    from gpu4pyscf.pbc.dft.numint import _GTOvalOpt
    from gpu4pyscf.pbc.df.ft_ao import libpbc
    from gpu4pyscf.lib.cupy_helper import contract

    import ctypes

    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    assert deriv <= 2
    if opt is None:
        opt = _GTOvalOpt(cell, kpts, deriv=deriv)
    else:
        assert kpts is opt.kpts

    bvkcell = opt.bvkcell
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    ngrids = len(coords)
    coords = cp.asarray(coords.T, order='C')
    bvk_ncells = opt.bvk_ncells

    nao = cell.nao
    sorted_cell = opt.sorted_cell
    # print(opt.gto_envs.ao_loc)
    ao_loc = cp.asarray(opt.gto_envs._env_ref_holder[3])
    print(ao_loc)
    
    print(f"{cell.nao = }")
    print(f"{sorted_cell.nao = }")
    print(f"{bvk_ncells = }")
    # assert 1 == 2

    out = cp.empty((comp, bvk_ncells, nao, ngrids))
    print(out.shape)
    assert 1 == 2

    drv = libpbc.PBCeval_gto_deriv
    err = drv(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(opt.gto_envs),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(bvk_ncells*nao), ctypes.c_int(bvkcell.nbas),
        ctypes.c_int(deriv), ctypes.c_int(cell.cart),
        ctypes.cast(opt.bas_rcut.data.ptr, ctypes.c_void_p)
    )
    if err != 0:
        raise RuntimeError('PBCeval_gto_deriv failed')

    out = out.reshape(comp, bvk_ncells, nao, ngrids)

    if bvk_ncells == 1: # gamma point
        out = out.transpose(1,0,3,2)
    else:
        bvk_ncells, nkpts = opt.expLk.shape
        expLk = opt.expLk.view(np.float64).reshape(bvk_ncells, nkpts, 2)
        out = contract('Lks,cLig->kcigs', expLk, out)
        out = out.view(np.complex128)[:,:,:,:,0].transpose(0,1,3,2)

    if deriv == 0:
        out = out[:,0]
    log.timer_debug2('eval_ao_kpts', *t0)
    return out


from pyscf.pbc import gto
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
pcell.basis = 'gth-szv-molopt-sr'
pcell.pseudo = "gth-hf-rev"
pcell.ke_cutoff = 200
# pcell.exp_to_discard = 0.1
pcell.verbose = 5
pcell.cart = True
pcell.build()

kpts = pcell.make_kpts([1, 1, 3])

from gpu4pyscf.pbc.scf import KRHF
mf = KRHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None

ni = mf.with_df._numint
gx = mf.with_df.grids.coords
ng = gx.shape[0]
ao_kpt = eval_ao_kpts(pcell, gx, kpts)
print(ao_kpt.shape)




