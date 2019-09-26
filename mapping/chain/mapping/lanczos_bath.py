from mapping.chain.mapping.base.sparse_bath import SparseDiagMapping
from mapping.tridiag.lanczos.diag import LanczosDiag
from mapping.tridiag.lanczos.diag_low_memory import LowMemLanczosDiag
import numpy as np


class LanczosBathMapping(SparseDiagMapping):
    def __init__(self, discretized_bath, max_nof_coefficients=100, low_memory=True, stable=False):
        """
            Star to chain mapping using the Lanczos algorithm on the bath only
        :param discretized_bath: Discretized bath object
        :param max_nof_coefficients: Maximum number of coefficients to be calculated
        :param low_memory: Uses a more memory efficient version of the algorithm, which may be slightly slower
        :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                       some stability problems encountered with Lanczos tridiagonalization
        """
        xi_buf = np.empty(discretized_bath.bufsize)
        v0_buf = np.empty(discretized_bath.bufsize)
        if not low_memory:
            tridiag = LanczosDiag(xi_buf, v0=v0_buf, max_cutoff=max_nof_coefficients, stable=stable)
        else:
            tridiag = LowMemLanczosDiag(xi_buf, v0=v0_buf, max_cutoff=max_nof_coefficients, stable=stable)
        super().__init__(xi_buf, v0_buf, discretized_bath, tridiag, max_nof_coefficients=max_nof_coefficients)
