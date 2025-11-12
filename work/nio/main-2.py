import cupy as cp, numpy as np
import pyscf, gpu4pyscf
import gpu4pyscf.pbc

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
phi_sol = ni.eval_ao(pcell, gx, kpts)
phi_sol = phi_sol.get()

from pyscf.pbc.scf import KRHF
mf = KRHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None

ni = mf.with_df._numint
gx = mf.with_df.grids.coords
ng = gx.shape[0]
phi_ref = ni.eval_ao(pcell, gx, kpts)
phi_ref = np.asarray(phi_ref)

print(f"{phi_sol.shape = }")
print(f"{phi_ref.shape = }")
err = np.abs(phi_sol - phi_ref).max()
print(f"{err = }")