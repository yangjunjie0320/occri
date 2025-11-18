import cupy as cp

device = cp.cuda.Device()
print(f"Using GPU device {device.id}")

mem_info = cp.cuda.runtime.memGetInfo()
print(f"Free memory: {mem_info[0] / (1024**3):.2f} GB")
print(f"Total memory: {mem_info[1] / (1024**3):.2f} GB")

import pyscf, gpu4pyscf
from pyscf.pbc import gto
import gpu4pyscf.pbc

import sys, pathlib
poscar_path = pathlib.Path("/resnick/groups/changroup/members/junjiey/occri/packages/")
poscar_path = poscar_path / "cuprate_parent_state_data/01_crystal_geometry/CCO/CCO-AFM-frac.vasp"
assert poscar_path.exists()

path = str(poscar_path)
print(f"poscar_path: {path}")

from libdmet.utils.iotools import read_poscar
pcell = read_poscar(path)
pcell.basis = 'gth-dzvp-molopt-sr'
pcell.pseudo = "gth-hf-rev"
pcell.ke_cutoff = 200
pcell.exp_to_discard = 0.1
pcell.verbose = 5
pcell.build()

kpts = pcell.make_kpts([2, 2, 1])

from gpu4pyscf.pbc.scf import KRHF, KUHF
mf = KUHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')

mf.kernel(dm0)
ene_ref = mf.e_tot

from fftdf import FFTDF
from fftdf import get_k_kpts_occri
mf.with_df = FFTDF(pcell, kpts, with_occri=True, use_gpu=True)
mf.with_df.blksize = 4
mf.kernel(dm0)
ene_sol = mf.e_tot

print("ene_ref = %12.6f" % ene_ref)
print("ene_sol = %12.6f" % ene_sol)
print("error   = %6.2e" % abs(ene_ref - ene_sol))

print("done")
