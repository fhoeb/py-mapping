"""
    Star to chain mapping using the scipy method hessenberg(using full storage matrices for the star coefficients)
"""
from mapping.chain.mapping.base.full import FullMapping
from mapping.tridiag.scipy_hessenberg.full import ScipyHessenberg
import numpy as np


class ScipyHessenbergMapping(FullMapping):
    def __init__(self, discretized_bath, max_nof_coefficients=100):
        """
            Star to chain mapping using the scipy method hessenberg.
        :param discretized_bath: Discretized bath object
        :param max_nof_coefficients: Maximum number of coefficients to be calculated
        """
        A_buf = np.zeros((discretized_bath.bufsize + 1, discretized_bath.bufsize + 1))
        tridiag = ScipyHessenberg(A_buf)
        super().__init__(A_buf, discretized_bath, tridiag, max_nof_coefficients=max_nof_coefficients)
