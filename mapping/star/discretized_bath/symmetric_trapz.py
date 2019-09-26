"""
    Discretized bath for the generation of direct symmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a vectorized version of the trapezoidal rule
"""
import numpy as np
from mapping.star.discretized_bath.base.symmetric import BaseDiscretizedSymmetricBath
from mapping.star.discretized_bath.intervals import get_symmetric_interval_points


class TrapzDiscretizedSymmetricBath(BaseDiscretizedSymmetricBath):
    def __init__(self, J, domain, max_nof_coefficients=100, interval_type='lin', **kwargs):
        """
            Generates direct discretization coefficients from a spectral density J, by computing the integrals:
            gamma_i = sqrt(int_i^i+1 J(x) dx)
            xi_i = int_i^i+1 J(x) * x dx/ gamma_i^2
        :param J: Spectral density. A function defined on 'domain', must be >0 in the inner part of domain
        :param domain: List/tuple of two elements for the left and right boundary of the domain of J. The
                       domain must contain 0.
        :param max_nof_coefficients: Size of the buffers which hold gamma and xi coefficients (maximum number of
                                     these coefficients that can be calculated)
        :param interval_type: see star.get_discretized_bath for an explanation of the available types
        :param kwargs: may contain 'ignore_zeros' If one gamma_i is numerically 0, the corresponding xi_i is also set 0,
                                                  default is False
        """
        assert not np.isinf(domain[0]) and not np.isinf(domain[1])
        if not domain[0] < 0 < domain[1]:
            print('Domain must contain 0!')
            raise AssertionError
        try:
            self.ignore_zeros = kwargs['ignore_zeros']
        except KeyError:
            self.ignore_zeros = False
        self.x_pts_p, self.x_pts_m, self.dx_p, self.dx_m = get_symmetric_interval_points(domain, max_nof_coefficients,
                                                                                         interval_type=interval_type,
                                                                                         get_spacing=True, **kwargs)
        self.interval_type = interval_type
        self.J = J
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)

    def compute_coefficients(self, stop_n):
        """
            Calculates the discretization coefficients up to stop_n (actually calculates 2*stop_n - self.next_n
            coefficients, since the indices are tailored for asymmetric discretizations)
        :param stop_n: Index up to which new coefficients are calculated
        """
        step_n = stop_n - self._next_n
        buffer_start = 2*self.next_n
        buffer_stop_pos = 2*self.next_n + step_n
        buffer_stop = 2*self._next_n + 2*step_n
        # Must invert the view, because for log-discretization the positive domain grid points are in inverted order
        x_pts_p = self.x_pts_p[::-1] if self.interval_type == 'log' else self.x_pts_p
        if isinstance(self.dx_p, np.ndarray):
            dx_p = self.dx_p[::-1] if self.interval_type == 'log' else self.dx_p
            dx_p_view = dx_p[self._next_n:stop_n]
        else:
            # dx is a float
            dx_p_view = self.dx_p
        if isinstance(self.dx_m, np.ndarray):
            dx_m_view = self.dx_m[self._next_n:stop_n]
        else:
            dx_m_view = self.dx_m
        # Positive part of the domain
        gamma_buf = self.gamma_buf[buffer_start:buffer_stop_pos]
        xi_buf = self.xi_buf[buffer_start:buffer_stop_pos]
        gamma_buf[:] = self.J(x_pts_p[self._next_n:stop_n]) + self.J(x_pts_p[self._next_n + 1:stop_n + 1])
        xi_buf[:] = x_pts_p[self._next_n:stop_n] * self.J(x_pts_p[self._next_n:stop_n]) + \
                    x_pts_p[self._next_n + 1:stop_n + 1] * self.J(x_pts_p[self._next_n + 1:stop_n + 1])
        if not self.ignore_zeros:
            xi_buf /= gamma_buf
        else:
            gamma_nonzero = np.nonzero(gamma_buf)
            gamma_zeros = np.nonzero(gamma_buf == 0)
            xi_buf[gamma_nonzero] /= gamma_buf[gamma_nonzero]
            xi_buf[gamma_zeros] = 0
        np.multiply(dx_p_view / 2, gamma_buf, out=gamma_buf)
        np.sqrt(gamma_buf, out=gamma_buf)
        # Negative part of the domain
        gamma_buf = self.gamma_buf[buffer_stop_pos:buffer_stop]
        xi_buf = self.xi_buf[buffer_stop_pos:buffer_stop]
        gamma_buf[:] = self.J(self.x_pts_m[self._next_n:stop_n]) + self.J(self.x_pts_m[self._next_n + 1:stop_n + 1])
        xi_buf[:] = self.x_pts_m[self._next_n:stop_n] * self.J(self.x_pts_m[self._next_n:stop_n]) + \
                    self.x_pts_m[self._next_n + 1:stop_n + 1] * self.J(self.x_pts_m[self._next_n + 1:stop_n + 1])
        if not self.ignore_zeros:
            xi_buf /= gamma_buf
        else:
            gamma_nonzero = np.nonzero(gamma_buf)
            gamma_zeros = np.nonzero(gamma_buf == 0)
            xi_buf[gamma_nonzero] /= gamma_buf[gamma_nonzero]
            xi_buf[gamma_zeros] = 0
        np.multiply(dx_m_view / 2, gamma_buf, out=gamma_buf)
        np.sqrt(gamma_buf, out=gamma_buf)
        self._set_next_n(stop_n)
