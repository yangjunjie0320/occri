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