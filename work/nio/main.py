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

kpts = pcell.make_kpts([1, 1, 1])

alph_label = ["0 Ni 3dx2-y2"]
beta_label = ["1 Ni 3dx2-y2"]
alph_ix = pcell.search_ao_label(alph_label)
beta_ix = pcell.search_ao_label(beta_label)

from gpu4pyscf.pbc.scf import KRHF, KUHF
mf = KUHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 50
mf.diis_space = 4
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')

dm0[0, :, alph_ix, alph_ix] *= 2.0
dm0[1, :, beta_ix, beta_ix] *= 2.0
dm0[0, :, beta_ix, beta_ix] *= 0.0
dm0[1, :, alph_ix, alph_ix] *= 0.0

from fftdf import FFTDF
mf.with_df = FFTDF(pcell, kpts, with_occri=True, use_gpu=True)
mf.with_df.blksize = 4
mf.kernel(dm0)
ene_sol = mf.e_tot

dm0 = mf.make_rdm1()

from gpu4pyscf.pbc.df.fft import FFTDF
mf.with_df = FFTDF(pcell, kpts)
mf.kernel(dm0)
ene_ref = mf.e_tot

print("ene_ref = %12.6f" % ene_ref)
print("ene_sol = %12.6f" % ene_sol)
print("error   = %6.2e" % abs(ene_ref - ene_sol))