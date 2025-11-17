import cupy as cp, numpy as np
cp.show_config()

import numpy as np
import pyscf, gpu4pyscf
from pyscf.pbc import gto
print(pyscf.__version__)
print(gpu4pyscf.__version__)

a = 10
basis = "gth-dzvp-molopt-sr"
pcell = gto.Cell()
pcell.atom = [["Ni", (a / 2, a / 2, a / 2)]]
pcell.a = np.diag([a, a, a])
pcell.basis = basis
pcell.pseudo = "gth-hf-rev"
pcell.verbose = 0
pcell.build()

from gpu4pyscf.pbc.dft.numint import _GTOvalOpt
kpts = pcell.make_kpts([1, 1, 3])
opt = _GTOvalOpt(pcell, kpts, deriv=0)
print("basis = %s, pcell.nao = %d, sorted_cell.nao = %d" % (basis, pcell.nao, opt.sorted_cell.nao))

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