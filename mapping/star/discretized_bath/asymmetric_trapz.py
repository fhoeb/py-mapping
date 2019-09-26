"""
    Discretized bath for the generation of direct asymmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a vectorized version of the trapezoidal rule
"""
import numpy as np
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.star.discretized_bath.intervals import get_asymmetric_interval_points


class TrapzDiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
    def __init__(self, J, domain, max_nof_coefficients=100, interval_type='lin', **kwargs):
        """
            Generates direct discretization coefficients from a spectral density J, by computing the integrals:
            gamma_i = sqrt(int_i^i+1 J(x) dx)
            xi_i = int_i^i+1 J(x) * x dx/ gamma_i^2
        :param J: Spectral density. A function defined on 'domain', must be >0 in the inner part of domain
        :param domain: List/tuple of two elements for the left and right boundary of the domain of J
        :param max_nof_coefficients: Size of the buffers which hold gamma and xi coefficients (maximum number of
                                     these coefficients that can be calculated)
        :param interval_type: see star.get_discretized_bath for an explanation of the available types
        :param kwargs: may contain 'ignore_zeros' If one gamma_i is numerically 0, the corresponding xi_i is also set 0,
                                                  default is False
        """
        assert not np.isinf(domain[1])
        self.x_pts, self.dx = get_asymmetric_interval_points(domain, max_nof_coefficients,
                                                             interval_type=interval_type,
                                                             get_spacing=True, **kwargs)
        try:
            self.ignore_zeros = kwargs['ignore_zeros']
        except KeyError:
            self.ignore_zeros = False
        self.J = J
        self.interval_type = interval_type
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)

    def compute_coefficients(self, stop_n):
        """
            Calculates the discretization coefficients for a given index n
        :param stop_n: Index of the coefficient
        """
        # Must invert the view, because for log-discretization the positive domain grid points are in inverted order
        x_pts = self.x_pts[::-1] if self.interval_type == 'log' else self.x_pts
        if isinstance(self.dx, np.ndarray):
            dx = self.dx[::-1] if self.interval_type == 'log' else self.dx
            dx_view = dx[self._next_n:stop_n]
        else:
            # dx is a float
            dx_view = self.dx
        gamma_buf = self.gamma_buf[self._next_n:stop_n]
        xi_buf = self.xi_buf[self._next_n:stop_n]
        gamma_buf[:] = self.J(x_pts[self._next_n:stop_n]) + self.J(x_pts[self._next_n + 1:stop_n + 1])
        xi_buf[:] = x_pts[self._next_n:stop_n] * self.J(x_pts[self._next_n:stop_n]) + \
                    x_pts[self._next_n + 1:stop_n + 1] * self.J(x_pts[self._next_n + 1:stop_n + 1])
        if not self.ignore_zeros:
            xi_buf /= gamma_buf
        else:
            gamma_nonzero = np.nonzero(gamma_buf)
            gamma_zeros = np.nonzero(gamma_buf == 0)
            xi_buf[gamma_nonzero] /= gamma_buf[gamma_nonzero]
            xi_buf[gamma_zeros] = 0
        np.multiply(dx_view/2, gamma_buf, out=gamma_buf)
        np.sqrt(gamma_buf, out=gamma_buf)
        self._set_next_n(stop_n)

