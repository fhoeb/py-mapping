"""
    Tridiagonalization class for the Householder method
"""
import numpy as np
from mapping.utils.residual import orth_residual


class Householder:
    def __init__(self, A, view=None):
        """
            Constructor
        :param A: Full storage real symmetric matrix to be tridiagonalized (is not overwritten)
        :param view: Integer, if present selects a sub-matrix of A starting from the top-left element,
                     up until the row/column n = view. This sub-matrix is then tridiagonalized instead of the full A
        """
        if A.shape[0] != A.shape[1]:
            print('Matrix must be square')
        self.dim = A.shape[0]
        if view is None:
            view = self.dim
        else:
            assert view <= self.dim
        # Underlying buffers with full dimensions
        self._A_buf = A[:, :]
        self._P_buf = np.empty((self.dim, self.dim), dtype=np.float64)
        self._v_buf = np.empty(self.dim, dtype=np.float64)
        self.n = view
        self.A = self._A_buf[:view, :view]
        self.P = self._P_buf[:view, :view]
        self.v = self._v_buf[:view]

    def update_view(self, view):
        """
            Updates the matrix view of A, which is tridiagonalized
        """
        assert view <= self.dim
        self.n = view
        self.A = self._A_buf[:view, :view]
        self.P = self._P_buf[:view, :view]
        self.v = self._v_buf[:view]

    def _core_loop(self, cutoff):
        """
            Performs the tridiagonalization up to 'cutoff' diagonal elements without computing the transformation matrix
        """
        # Init
        alpha = - np.sign(self.A[1, 0]) * np.sqrt(np.sum(np.square(self.A[1:, 0])))
        r = np.sqrt(1 / 2 * (alpha ** 2 - self.A[1, 0] * alpha))
        self.v[0] = 0
        self.v[1] = (self.A[1, 0] - alpha) / (2 * r)
        self.v[2:] = self.A[2:, 0] / (2 * r)
        self.P[:, :] = - 2 * np.outer(self.v, self.v)
        self.P[np.diag_indices_from(self.P)] += 1
        A = self.P @ self.A @ self.P
        # Iteration:
        for k in range(1, cutoff - 1):
            alpha = - np.sign(A[k + 1, k]) * np.sqrt(np.sum(np.square(A[k + 1:, k])))
            r = np.sqrt(1 / 2 * (alpha ** 2 - A[k + 1, k] * alpha))
            self.v[:k + 1].fill(0)
            self.v[k + 1] = (A[k + 1, k] - alpha) / (2 * r)
            self.v[k + 2:] = A[k + 2:, k] / (2 * r)
            self.P[:, :] = - 2 * np.outer(self.v, self.v)
            self.P[np.diag_indices_from(self.P)] += 1
            np.matmul(self.P, A, out=A)
            np.matmul(A, self.P, out=A)
        diag = np.diag(A)[:cutoff].copy()
        offdiag = np.diag(A, k=1)[:cutoff - 1].copy()
        return diag, offdiag

    def _core_loop_with_trafo(self, cutoff):
        """
            Performs the tridiagonalization up to 'cutoff' diagonal elements, while also
            computing the transformation matrix
        """
        # Init
        alpha = - np.sign(self.A[1, 0]) * np.sqrt(np.sum(np.square(self.A[1:, 0])))
        r = np.sqrt(1 / 2 * (alpha ** 2 - self.A[1, 0] * alpha))
        self.v[0] = 0
        self.v[1] = (self.A[1, 0] - alpha) / (2 * r)
        self.v[2:] = self.A[2:, 0] / (2 * r)
        self.P[:, :] = - 2 * np.outer(self.v, self.v)
        self.P[np.diag_indices_from(self.P)] += 1
        A = self.P @ self.A @ self.P
        # Matrix Q stores full transformation
        Q = self.P.copy()
        # Iteration:
        for k in range(1, cutoff - 1):
            alpha = - np.sign(A[k + 1, k]) * np.sqrt(np.sum(np.square(A[k + 1:, k])))
            r = np.sqrt(1 / 2 * (alpha ** 2 - A[k + 1, k] * alpha))
            self.v[:k + 1].fill(0)
            self.v[k + 1] = (A[k + 1, k] - alpha) / (2 * r)
            self.v[k + 2:] = A[k + 2:, k] / (2 * r)
            self.P[:, :] = - 2 * np.outer(self.v, self.v)
            self.P[np.diag_indices_from(self.P)] += 1
            np.matmul(self.P, A, out=A)
            np.matmul(A, self.P, out=A)
            np.matmul(Q, self.P, out=Q)
        diag = np.diag(A)[:cutoff].copy()
        offdiag = np.diag(A, k=1)[:cutoff - 1].copy()
        return diag, offdiag, Q

    def _make_positive(self, diag, offdiag, trafo=None):
        """
            Makes the offdiagonal elements of the tridiagonal positive.
            If no trafo matrix is passed, the absolute value of the offdiagonal elements is take, as there always
            exists a transformation that would perform the correct transformation
        :param diag: Diagonal elements of the triiagonal matrix
        :param offdiag: Offdiagonal elements of the tridiagonal matrix
        :param trafo: Transformation matrix (if not present see function description). If passed, the signs of the
                      orthonormal column vectors are flipped to make the corresponding offdiagonal elements positive
        :return: diagonal elements of the triiagonal matrix, strictly positive offdiagonals,
                 the corresponding transformation matrix (if trafo was passed and not None)
        """
        if trafo is None:
            return diag, np.abs(offdiag)
        else:
            buffer = offdiag.copy()
            index = 0
            while index < len(buffer):
                if buffer[index] < 0:
                    trafo[:, index + 1] = -trafo[:, index + 1]
                    A_pos = trafo.T @ self.A @ trafo
                    buffer = np.diag(A_pos, k=1)
                index += 1
            # Test if it worked:
            assert np.all(buffer >= 0)
            return diag, np.abs(offdiag), trafo

    def get_tridiagonal(self, cutoff=None, residual=False, get_trafo=False, positive=True):
        """
            Main interface. Computes the tridiagonal elements of the matrix A and returns them
        :param cutoff: How many diagonal elements for the resulting tridiagonal elements should be computed
                       If left unchanged the entire matrix is tridiagonalized
        :param residual: If set True computes a residual. See utils.residual.orth_residual for details
        :param get_trafo: If set True computes and returns the transformaion matrix
        :param positive: If set True ensures that the offdiagonal elements are chosen to be positive
        :return: diag (diagonal elements of the computed tridiagonal matrix), offdiag (offdiagonal elements
                 of the computed tridiagonal matrix), info dict with keys 'trafo' and 'res', which
                 contain the corresponding transformation matrix and residual of the computation respectively
        """
        if cutoff is None:
            cutoff = self.n
        else:
            assert 0 < cutoff <= self.n
        info = dict()
        info['trafo'] = None
        info['res'] = None
        if residual or get_trafo:
            diag, offdiag, Q = self._core_loop_with_trafo(cutoff)
            if positive and get_trafo:
                diag, offdiag, Q = self._make_positive(diag, offdiag, trafo=Q)
            else:
                diag, offdiag = self._make_positive(diag, offdiag)
            if get_trafo:
                info['trafo'] = Q
            if residual:
                info['res'] = orth_residual(Q[:, :cutoff])
        else:
            diag, offdiag = self._core_loop(cutoff)
            if positive:
                diag, offdiag = self._make_positive(diag, offdiag)
        return diag, offdiag, info
