import numpy as np
from mapping.utils.residual import orth_residual
from math import fsum


class LowMemLanczosSpecialSparse:
    def __init__(self, gamma, xi, v0=None, view=None, max_cutoff=None, stable=False):
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
        self._V_buf = np.empty((self.dim, 2), dtype=np.float64, order='F')
        self._w_buf = np.empty(self.dim, dtype=np.float64)
        self._alpha_buf = np.empty(self.max_cutoff, dtype=np.float64)
        self._beta_buf = np.empty(self.max_cutoff, dtype=np.float64)
        self.n = view
        self.gamma = self._gamma_buf[:view-1]
        self.xi = self._xi_buf[:view-1]
        self.v0 = self._v0_buf[:view]
        self.V = self._V_buf[:view, :2]
        self.w = self._w_buf[:view]
        self.alpha = self._alpha_buf
        self.beta = self._beta_buf

    def update_view(self, view):
        assert view <= self.dim
        self.n = view
        self.gamma = self._gamma_buf[:view-1]
        self.xi = self._xi_buf[:view-1]
        self.v0 = self._v0_buf[:view]
        self.V = self._V_buf[:view, :2]
        self.w = self._w_buf[:view]

    def _dgemv(self, v):
        self.w[0] = self.gamma.dot(v[1:])
        self.w[1:] = self.gamma * v[0] + self.xi*v[1:]

    def _core_loop(self, cutoff):
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
                    raise AssertionError
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, 1])
                self._dgemv(self.V[:, 1])
                self.alpha[i] = self.w.dot(self.V[:, 1])
                self.w -= self.alpha[i] * self.V[:, 1] + self.beta[i] * self.V[:, 0]
                self.V[:, 0] = self.V[:, 1]
        else:
            for i in range(1, cutoff):
                self.beta[i] = np.sqrt(fsum(np.square(self.w)))
                if self.beta[i] == 0:
                    raise AssertionError
                else:
                    np.multiply(1/self.beta[i], self.w, out=self.V[:, 1])
                self._dgemv(self.V[:, 1])
                self.alpha[i] = self.w.dot(self.V[:, 1])
                self.w -= self.alpha[i] * self.V[:, 1] + self.beta[i] * self.V[:, 0]
                self.V[:, 0] = self.V[:, 1]
        return self.alpha[:cutoff].copy(), self.beta[1:cutoff].copy()

    def _core_loop_with_trafo(self, cutoff):
        V = np.empty((self.V.shape[0], cutoff), dtype=np.float64, order='F')
        # Initial step:
        V[:, 0] = self.v0
        self._dgemv(V[:, 0])
        self.alpha[0] = self.w.dot(V[:, 0])
        self.w[:] -= self.alpha[0] * V[:, 0]
        # Core loop:
        if not self.stable:
            for i in range(1, cutoff):
                self.beta[i] = np.linalg.norm(self.w)
                if self.beta[i] == 0:
                    raise AssertionError
                else:
                    np.multiply(1/self.beta[i], self.w, out=V[:, i])
                self._dgemv(V[:, i])
                self.alpha[i] = self.w.dot(V[:, i])
                self.w[:] -= self.alpha[i] * V[:, i] + self.beta[i] * V[:, i - 1]
        else:
            for i in range(1, cutoff):
                self.beta[i] = np.sqrt(fsum(np.square(self.w)))
                if self.beta[i] == 0:
                    raise AssertionError
                else:
                    np.multiply(1/self.beta[i], self.w, out=V[:, i])
                self._dgemv(V[:, i])
                self.alpha[i] = self.w.dot(V[:, i])
                self.w[:] -= self.alpha[i] * V[:, i] + self.beta[i] * V[:, i - 1]
        return self.alpha[:cutoff].copy(), self.beta[1:cutoff].copy(), V

    def get_tridiagonal(self, cutoff=None, residual=False, get_trafo=False):
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
