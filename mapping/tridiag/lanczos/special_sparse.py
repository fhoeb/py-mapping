"""
    Tridiagonalization class for the Lanczos algorithm for special matrices of the form:
        / 0 s1 s2 ..  \\
        |s1 d1 0  ..  |
        |s2 0 d2  ..  |
        \\ : : :      /
    gamma contains the si (the elements of the top and left edge of the matrix) and
    xi contains the diagonal elements starting from element [1, 1] as 1d numpy arrays.
"""
import numpy as np
from mapping.utils.orth import get_orthogonal_vector
from mapping.utils.residual import orth_residual
from math import fsum


class LanczosSpecialSparse:
    def __init__(self, gamma, xi, v0=None, view=None, max_cutoff=None, stable=False):
        """
            Constructor
        :param xi: Diagonal elements of the matrix
        :param gamma: Top/left edge elements of the matrix (save for A[0,0])
        :param v0: Starting (normalized) vector for the tridiagonalization, default is (1 0 ... 0).T
        :param view: Integer, if present selects a sub-matrix of A starting from the top-left element,
                     up until the row/column n = view. This sub-matrix is then tridiagonalized instead of the full A
        :param max_cutoff: Maximum amount of diagonal elements (of the tridiagonalized matrix) that will be computed
                           Can be set to save some memory. If left unchanges, will be set to the dimension of A
        :param stable: Uses a stable summation algorithm (WARNING: slow)
        """
        if len(gamma) != len(xi):
            print('Matrix must be square')
        self.dim = len(gamma) + 1
        self.stable = stable
        if view is None:
            view = self.dim
        else:
            assert view <= self.dim
        if max_cutoff is None:
            max_cutoff = self.dim
        else:
            assert max_cutoff <= self.dim
        if v0 is None:
            v0 = np.zeros(self.dim)
            v0[0] = 1
        self.max_cutoff = max_cutoff
        # Underlying buffers with full dimensions
        self._gamma_buf = gamma[:]
        self._xi_buf = xi[:]
        self._v0_buf = v0[:]
        self._V_buf = np.empty((self.dim, self.dim), dtype=np.float64, order='F')
        self._w_buf = np.empty(self.dim, dtype=np.float64)
        self._alpha_buf = np.empty(self.max_cutoff, dtype=np.float64)
        self._beta_buf = np.empty(self.max_cutoff, dtype=np.float64)
        self.n = view
        self.gamma = self._gamma_buf[:view-1]
        self.xi = self._xi_buf[:view-1]
        self.v0 = self._v0_buf[:view]
        self.V = self._V_buf[:view, :view]
        self.w = self._w_buf[:view]
        self.alpha = self._alpha_buf
        self.beta = self._beta_buf

    def update_view(self, view):
        """
            Updates the matrix view of A, which is tridiagonalized
        """
        assert view <= self.dim
        self.n = view
        self.gamma = self._gamma_buf[:view-1]
        self.xi = self._xi_buf[:view-1]
        self.v0 = self._v0_buf[:view]
        self.V = self._V_buf[:view, :view]
        self.w = self._w_buf[:view]

    def _dgemv(self, v):
        """
            Custom row major matrix vector product between A and a vector v
            Updates the w buffer
        :param v: Vector with which the matrix vector product is taken
        """
        self.w[0] = self.gamma.dot(v[1:])
        self.w[1:] = self.gamma*v[0] + self.xi*v[1:]

    def _core_loop(self, cutoff):
        """
            Performs the tridiagonalization up to 'cutoff' diagonal elements without computing the transformation matrix
        """
        # Initial step:
        self.V[:, 0] = self.v0
        self._dgemv(self.V[:, 0])
        self.alpha[0] = self.w.dot(self.V[:, 0])
        self.w -= self.alpha[0] * self.V[:, 0]
        # Core loop:
        if not self.stable:
            for i in range(1, cutoff):
                self.beta[i] = np.linalg.norm(self.w)
                if self.beta[i] == 0:
                    self.V[:, i] = get_orthogonal_vector(self.V[:, :i])
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, i])
                self._dgemv(self.V[:, i])
                self.alpha[i] = self.w.dot(self.V[:, i])
                self.w -= self.alpha[i] * self.V[:, i] + self.beta[i] * self.V[:, i - 1]
        else:
            for i in range(1, cutoff):
                self.beta[i] = np.sqrt(fsum(np.square(self.w)))
                if self.beta[i] == 0:
                    self.V[:, i] = get_orthogonal_vector(self.V[:, :i])
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, i])
                self._dgemv(self.V[:, i])
                self.alpha[i] = self.w.dot(self.V[:, i])
                self.w -= self.alpha[i] * self.V[:, i] + self.beta[i] * self.V[:, i - 1]
        return self.alpha[:cutoff].copy(), self.beta[1:cutoff].copy()

    def _core_loop_with_trafo(self, cutoff):
        """
            Performs the tridiagonalization up to 'cutoff' diagonal elements, while also
            computing the transformation matrix
        """
        # Initial step:
        self.V[:, 0] = self.v0
        self._dgemv(self.V[:, 0])
        self.alpha[0] = self.w.dot(self.V[:, 0])
        self.w -= self.alpha[0] * self.V[:, 0]
        # Core loop:
        if not self.stable:
            for i in range(1, cutoff):
                self.beta[i] = np.linalg.norm(self.w)
                if self.beta[i] == 0:
                    self.V[:, i] = get_orthogonal_vector(self.V[:, :i])
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, i])
                self._dgemv(self.V[:, i])
                self.alpha[i] = self.w.dot(self.V[:, i])
                self.w -= self.alpha[i] * self.V[:, i] + self.beta[i] * self.V[:, i-1]
        else:
            for i in range(1, cutoff):
                self.beta[i] = np.sqrt(fsum(np.square(self.w)))
                if self.beta[i] == 0:
                    self.V[:, i] = get_orthogonal_vector(self.V[:, :i])
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, i])
                self._dgemv(self.V[:, i])
                self.alpha[i] = self.w.dot(self.V[:, i])
                self.w -= self.alpha[i] * self.V[:, i] + self.beta[i] * self.V[:, i-1]
        return self.alpha[:cutoff].copy(), self.beta[1:cutoff].copy(), self.V[:, :cutoff]

    def get_tridiagonal(self, cutoff=None, residual=False, get_trafo=False):
        """
            Main interface. Computes the tridiagonal elements of the matrix A and returns them
        :param cutoff: How many diagonal elements for the resulting tridiagonal elements should be computed
                       If left unchanged the entire matrix is tridiagonalized
        :param residual: If set True computes a residual. See utils.residual.orth_residual for details
        :param get_trafo: If set True computes and returns the transformaion matrix
        :return: diag (diagonal elements of the computed tridiagonal matrix), offdiag (offdiagonal elements
                 of the computed tridiagonal matrix), info dict with keys 'trafo' and 'res', which
                 contain the corresponding transformation matrix and residual of the computation respectively
        """
        if cutoff is None:
            cutoff = self.max_cutoff
        else:
            assert 0 < cutoff <= self.max_cutoff
        info = dict()
        info['trafo'] = None
        info['res'] = None
        if residual or get_trafo:
            diag, offdiag, V = self._core_loop_with_trafo(cutoff)
            if get_trafo:
                info['trafo'] = V
            if residual:
                info['res'] = orth_residual(V)
        else:
            diag, offdiag = self._core_loop(cutoff)
        return diag, offdiag, info