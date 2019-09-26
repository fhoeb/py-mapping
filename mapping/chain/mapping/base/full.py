"""
    Mapping class, which maps star to chain using the a dummy system and the full bath. For tridiagonal
    solvers, which use full storage matrices
"""
from mapping.chain.mapping.base.base import BaseMapping
import numpy as np


class FullMapping(BaseMapping):
    def __init__(self, A_buf, discretized_bath, tridiag, max_nof_coefficients=100):
        """
            Mapping for full storage star coefficient matrices, which include the bath and a dummy system.
        :param A_buf: Full storage Matrix buffer big enough to fit max_nof_coefficients of the
                      discretized bath
        :param discretized_bath: Discretized bath object
        :param tridiag: Tridiagonal solver
        :param max_nof_coefficients: Maximum number of chain coefficients to be calculated
        """
        self.A_buf = A_buf
        super().__init__(discretized_bath, tridiag, 'full', max_nof_coefficients)

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
            drows, dcols = np.diag_indices_from(self.A_buf)
            self.A_buf[drows[1:ncap + 1], dcols[1:ncap + 1]] = xi[sorted_indices]
            self.A_buf[0, 1:ncap + 1] = gamma[sorted_indices]
            self.A_buf[1:ncap + 1, 0] = self.A_buf[0, 1:ncap+1]
        else:
            self.A_buf[0, 1:ncap+1] = gamma
            self.A_buf[1:ncap+1, 0] = gamma
            drows, dcols = np.diag_indices_from(self.A_buf)
            self.A_buf[drows[1:ncap+1], dcols[1:ncap+1]] = xi
        self.tridiag.update_view(ncap+1)

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
            drows, dcols = np.diag_indices_from(self.A_buf)
            self.A_buf[drows[1:2 * ncap + 1], dcols[1:2 * ncap + 1]] = xi[sorted_indices]
            self.A_buf[0, 1:2 * ncap + 1] = gamma[sorted_indices]
            self.A_buf[1:2 * ncap + 1, 0] = self.A_buf[0, 1:2 * ncap + 1]
        else:
            self.A_buf[0, 1:2*ncap+1] = gamma
            self.A_buf[1:2*ncap+1, 0] = gamma
            drows, dcols = np.diag_indices_from(self.A_buf)
            self.A_buf[drows[1:2*ncap+1], dcols[1:2*ncap+1]] = xi
        self.tridiag.update_view(2*ncap+1)
