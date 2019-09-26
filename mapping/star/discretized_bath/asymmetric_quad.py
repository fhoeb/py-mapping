"""
    Discretized bath for the generation of direct asymmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a vectorized version of the Gauss-Legendre quadrature
    with fixed order
"""
from mapping.utils.gl_quad import compute_quadrature_integrals
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.star.discretized_bath.intervals import get_asymmetric_interval_points
import numpy as np


class QuadDiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
    """
        Generation of coefficients for linear discretization of bosonic baths
    """
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
                                   'quad': Order of the qudrature to be used may be any value between n=2 and
                                           n=64 (including those two). Default is 30
        """
        assert not np.isinf(domain[1])
        x_pts, self.dx = get_asymmetric_interval_points(domain, max_nof_coefficients, interval_type=interval_type,
                                                        get_spacing=True, **kwargs)
        self.mid = (x_pts[:-1] + x_pts[1:])/2
        self.J = J
        self.interval_type = interval_type
        try:
            self.ignore_zeros = kwargs['ignore_zeros']
        except KeyError:
            self.ignore_zeros = False
        try:
            self.order = kwargs['order']
        except KeyError:
            self.order = 30
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)

    def compute_coefficients(self, stop_n):
        """
            Calculates the discretization coefficients up to stop_n
        :param stop_n: Index up to which new coefficients are calculated
        """
        # Must invert the view, because for log-discretization the positive domain grid points are in inverted order
        mid = self.mid[::-1] if self.interval_type == 'log' else self.mid
        if isinstance(self.dx, np.ndarray):
            dx = self.dx[::-1] if self.interval_type == 'log' else self.dx
            dx_view = dx[self._next_n:stop_n]
        else:
            # dx is a float
            dx_view = self.dx
        compute_quadrature_integrals(self.J, mid[self._next_n:stop_n], dx_view,
                                     self.gamma_buf[self._next_n:stop_n],  self.xi_buf[self._next_n:stop_n],
                                     ignore_zeros=self.ignore_zeros, order=self.order)
        self._set_next_n(stop_n)
