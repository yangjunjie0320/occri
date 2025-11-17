def FFTDF(use_cpu=True, use_gpu=False, save_memory=False, version=None):
    assert not (use_cpu and use_gpu)
    assert use_cpu or use_gpu

    df_obj = None
    if version == "pyscf":
        import pyscf, gpu4pyscf
        assert not save_memory

        from pyscf.pbc.df.fftdf import FFTDF
        df_obj = FFTDF(mesh=mesh, precision=precision)
        if use_gpu:
            import gpu4pyscf.pbc.df.fftdf
            df_obj = df_obj.to_gpu()

        assert df_obj is not None
        return df_obj

    import 

from gpu4pyscf.pbc.df import fft
class OccRI(fft.FFTDF):
    blockdim = 240


    def __init__(self, cell, kpts=None):
        from gpu4pyscf.pbc.dft import numint
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
        self.kpts = kpts

        # The following attributes are not input options.
        # self.exxdiv has no effects. It was set in the get_k_kpts function to
        # mimic the KRHF/KUHF object in the call to tools.get_coulG.
        self.exxdiv = None
        self._numint = numint.KNumInt()
        self._rsh_df = {}  # Range separated Coulomb DF objects

    def dump_flags(self):
        return self._flags