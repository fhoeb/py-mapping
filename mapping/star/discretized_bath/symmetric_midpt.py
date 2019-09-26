"""
    Discretized bath for the generation of direct symmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a vectorized version of the midpoint rule
"""
import numpy as np
from mapping.star.discretized_bath.base.symmetric import BaseDiscretizedSymmetricBath
from mapping.star.discretized_bath.intervals import get_symmetric_interval_points


class MidptDiscretizedSymmetricBath(BaseDiscretizedSymmetricBath):
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
        :param kwargs: dummy argument for compatibility with other discretized baths
        """
        assert not np.isinf(domain[0]) and not np.isinf(domain[1])
        if not domain[0] < 0 < domain[1]:
            print('Domain must contain 0!')
            raise AssertionError
        x_pts_p, x_pts_m, self.dx_p, self.dx_m = get_symmetric_interval_points(domain, max_nof_coefficients,
                                                                               interval_type=interval_type,
                                                                               get_spacing=True, **kwargs)
        self.mid_p = (x_pts_p[:-1] + x_pts_p[1:]) / 2
        self.mid_m = (x_pts_m[:-1] + x_pts_m[1:]) / 2
        self.J = J
        self.interval_type = interval_type
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
        mid_p = self.mid_p[::-1] if self.interval_type == 'log' else self.mid_p
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
        self.gamma_buf[buffer_start:buffer_stop_pos] = np.sqrt(self.J(mid_p[self.next_n:stop_n]) *
                                                               dx_p_view)
        self.xi_buf[buffer_start:buffer_stop_pos] = mid_p[self.next_n:stop_n]
        self.gamma_buf[buffer_stop_pos:buffer_stop] = np.sqrt(self.J(self.mid_m[self.next_n:stop_n]) *
                                                              dx_m_view)
        self.xi_buf[buffer_stop_pos:buffer_stop] = self.mid_m[self.next_n:stop_n]
        self._set_next_n(stop_n)
