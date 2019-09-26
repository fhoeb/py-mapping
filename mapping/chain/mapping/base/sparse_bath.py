"""
    Mapping class, which maps star to chain using only the bath. For tridiagonal
    solvers, which tridiagonalize diagonal matrices
"""
from mapping.chain.mapping.base.base import BaseMapping
import numpy as np


class SparseDiagMapping(BaseMapping):
    def __init__(self, xi_buf, v0_buf, discretized_bath, tridiag, max_nof_coefficients=100):
        """
            Mapping for diagonal star bath coefficient matrices, which include only the bath
        :param xi_buf: Numpy array of the diagonal elements (the bathe energies)
        :param v0_buf: Numpy array of the normalized couplings (the system-bath couplings), which is
                       used as initial vector for the Lanczos tridiagonalization
        :param discretized_bath: Discretized bath object
        :param tridiag: Tridiagonal solver
        :param max_nof_coefficients: Maimum number of chain coefficients to be calculated
        """
        self.xi_buf = xi_buf
        self.v0_buf = v0_buf
        super().__init__(discretized_bath, tridiag, 'bath', max_nof_coefficients)

    def _asymmetric_update(self, ncap):
        """
            Updates to use 2*ncap discretized bath coefficients from the star (the factor of two to
            include symmetry)
        :param ncap: Accuracy parameter/ Number of star coefficients to update the tridiaggonal solver with
        :return:
        """
        self.discretized_bath.compute_until(ncap)
        gamma = self.discretized_bath.gamma_buf[:ncap]
        xi = self.discretized_bath.xi_buf[:ncap]
        if self.sorting.sort is not None:
            sorted_indices = self.sorting.sort(gamma, xi)
            self.xi_buf[:ncap] = xi[sorted_indices]
            self.v0_buf[:ncap] = gamma[sorted_indices]/np.linalg.norm(gamma)
        else:
            self.xi_buf[:ncap] = xi
            self.v0_buf[:ncap] = gamma/np.linalg.norm(gamma)
        self.tridiag.update_view(ncap)

    def _symmetric_update(self, ncap):
        """
            Updates to use ncap discretized bath coefficients from the star
        :param ncap: Accuracy parameter/ Number of star coefficients to update the tridiaggonal solver with
        :return:
        """
        self.discretized_bath.compute_until(ncap)
        gamma = self.discretized_bath.gamma_buf[:2*ncap]
        xi = self.discretized_bath.xi_buf[:2*ncap]
        if self.sorting.sort is not None:
            sorted_indices = self.sorting.sort(gamma, xi)
            self.xi_buf[:2*ncap] = xi[sorted_indices]
            self.v0_buf[:2*ncap] = gamma[sorted_indices]/np.linalg.norm(gamma)
        else:
            self.xi_buf[:2*ncap] = xi
            self.v0_buf[:2*ncap] = gamma
            self.v0_buf /= np.linalg.norm(self.v0_buf)
        self.tridiag.update_view(2*ncap)
