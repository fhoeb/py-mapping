"""
    Star to chain mapping using Householder transformations (using full storage matrices for the star coefficients)
"""
from mapping.chain.mapping.base.full import FullMapping
from mapping.tridiag.householder.full import Householder
import numpy as np


class HouseholderMapping(FullMapping):
    def __init__(self, discretized_bath, max_nof_coefficients=100):
        """
            Star to chain mapping using Householder transformations
        :param discretized_bath: Discretized bath object
        :param max_nof_coefficients: Maximum number of coefficients to be calculated
        """
        A_buf = np.zeros((discretized_bath.bufsize + 1, discretized_bath.bufsize + 1))
        tridiag = Householder(A_buf)
        super().__init__(A_buf, discretized_bath, tridiag, max_nof_coefficients=max_nof_coefficients)
