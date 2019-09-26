"""
    Mapping class, which maps star to chain using the a dummy system and the full bath. For tridiagonal
    solvers, which use the special sparse form of the star coefficient matrix
"""
from mapping.chain.mapping.base.base import BaseMapping


class SpecialSparseMapping(BaseMapping):
    def __init__(self, gamma_buf, xi_buf, discretized_bath, tridiag, max_nof_coefficients=100):
        """
            Mapping for sparse star coefficient matrices, which include the bath and a dummy system.
        :param gamma_buf: Coupling coefficients in the star system (on the edges of the matrix) as array
        :param xi_buf: Energies in the star system (on the diagonal) as array
        :param discretized_bath: Discretized bath object
        :param tridiag: Tridiagonal solver
        :param max_nof_coefficients: Maximum number of chain coefficients to be calculated
        """
        self.gamma_buf = gamma_buf
        self.xi_buf = xi_buf
        super().__init__(discretized_bath, tridiag, 'full', max_nof_coefficients)

    def _asymmetric_update(self, ncap):
        self.discretized_bath.compute_until(ncap)
        gamma = self.discretized_bath.gamma_buf[:ncap]
        xi = self.discretized_bath.xi_buf[:ncap]
        if self.sorting.sort is not None:
            sorted_indices = self.sorting.sort(gamma, xi)
            self.xi_buf[:ncap] = xi[sorted_indices]
            self.gamma_buf[:ncap] = gamma[sorted_indices]
        else:
            self.gamma_buf[:ncap] = gamma
            self.xi_buf[:ncap] = xi
        self.tridiag.update_view(ncap+1)

    def _symmetric_update(self, ncap):
        self.discretized_bath.compute_until(ncap)
        gamma = self.discretized_bath.gamma_buf[:2*ncap]
        xi = self.discretized_bath.xi_buf[:2*ncap]
        if self.sorting.sort is not None:
            sorted_indices = self.sorting.sort(gamma, xi)
            self.xi_buf[:2 * ncap] = xi[sorted_indices]
            self.gamma_buf[:2 * ncap] = gamma[sorted_indices]
        else:
            self.gamma_buf[:2*ncap] = gamma
            self.xi_buf[:2*ncap] = xi
        self.tridiag.update_view(2*ncap+1)
