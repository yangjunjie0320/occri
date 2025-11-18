int build_vR_dot_dm(const complex* mo1T, const complex* mo2T, const complex* coulg, complex* buffer, complex* vr_dm, int ngrids, int nmo1, int nmo2, int blksize) {
    for (int i0 = 0; i0 < nmo1; i0 += blksize) {
        int i1 = min(i0 + blksize, nmo1);
        for (int j0 = 0; j0 < nmo2; j0 += blksize) {
            int j1 = min(j0 + blksize, nmo2);
            for (int k = 0; k < ngrids; k++) {
                complex* rhoR = buffer;
                // rhoR = mo1T[i].reshape * mo2T[j0:j1]
