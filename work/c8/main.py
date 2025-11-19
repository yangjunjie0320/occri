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
pcell.exp_to_discard = 0.1
pcell.build()

kpts = pcell.make_kpts([2, 2, 2])

from gpu4pyscf.pbc.scf import KRHF, KUHF
mf = KRHF(pcell, kpts)
mf.verbose = 5
mf.conv_tol = 1e-6
mf.max_cycle = 10
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')

mf.kernel(dm0)
ene_ref = mf.e_tot

from fftdf import FFTDF
mf.with_df = FFTDF(pcell, kpts, with_occri=True, use_gpu=True)
mf.with_df.blksize = 4
mf.kernel(dm0)
ene_sol = mf.e_tot

print("ene_ref = %12.6f" % ene_ref)
print("ene_sol = %12.6f" % ene_sol)
print("error   = %6.2e" % abs(ene_ref - ene_sol))

print("done")
