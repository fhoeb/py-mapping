"""
    Star to chain mapping using the Lanczos algorithm (using full storage matrices for the star coefficients)
"""
from mapping.chain.mapping.base.special_sparse import SpecialSparseMapping
from mapping.tridiag.lanczos.special_sparse import LanczosSpecialSparse
from mapping.tridiag.lanczos.special_sparse_low_memory import LowMemLanczosSpecialSparse
import numpy as np


class LanczosMapping(SpecialSparseMapping):
    def __init__(self, discretized_bath, max_nof_coefficients=100, low_memory=True, stable=False):
        """
            Star to chain mapping using the Lanczos algorithm on the combination of dummy system and bath
        :param discretized_bath: Discretized bath object
        :param max_nof_coefficients: Maximum number of coefficients to be calculated
        :param low_memory: Uses a more memory efficient version of the algorithm, which may be slightly slower
        :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                       some stability problems encountered with Lanczos tridiagonalization
        """
        gamma_buf = np.empty(discretized_bath.bufsize)
        xi_buf = np.empty(discretized_bath.bufsize)
        v0_buf = np.zeros(discretized_bath.bufsize+1)
        v0_buf[0] = 1
        if not low_memory:
            tridiag = LanczosSpecialSparse(gamma_buf, xi_buf, v0=v0_buf, max_cutoff=max_nof_coefficients, stable=stable)
        else:
            tridiag = LowMemLanczosSpecialSparse(gamma_buf, xi_buf, v0=v0_buf, max_cutoff=max_nof_coefficients,
                                                 stable=stable)
        super().__init__(gamma_buf, xi_buf, discretized_bath, tridiag, max_nof_coefficients=max_nof_coefficients)
