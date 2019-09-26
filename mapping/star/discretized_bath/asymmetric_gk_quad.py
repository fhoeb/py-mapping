"""
    Discretized bath for the generation of direct asymmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a vectorized version of the (adaptive) Gauss-Kronrod quadrature
"""
from mapping.utils.gk_quad import compute_gk_quadrature_integrals
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.star.discretized_bath.intervals import get_asymmetric_interval_points
from mapping.utils.integration_defaults import default_epsabs, default_epsrel
import numpy as np


class GKQuadDiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
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
                                   'epsabs': absolute tolerance for the integration, default is 1e-11
                                   'epsrel': relative tolerance for the integration, default is 1e-11
        """
        assert not np.isinf(domain[1])
        x_pts, self.dx = get_asymmetric_interval_points(domain, max_nof_coefficients, interval_type=interval_type,
                                                        get_spacing=True, **kwargs)
        self.mid = (x_pts[:-1] + x_pts[1:])/2
        self.J = J
        try:
            self.ignore_zeros = kwargs['ignore_zeros']
        except KeyError:
            self.ignore_zeros = False
        try:
            self.epsabs = kwargs['epsabs']
        except KeyError:
            self.epsabs = default_epsabs
        try:
            self.epsrel = kwargs['epsrel']
        except KeyError:
            self.epsrel = default_epsrel
        self.interval_type = interval_type
        self.comp_gamma_buf = np.empty(max_nof_coefficients)
        self.comp_xi_buf = np.empty(max_nof_coefficients)
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
        compute_gk_quadrature_integrals(self.J, mid[self._next_n:stop_n], dx_view,
                                        self.comp_gamma_buf[self._next_n:stop_n],
                                        self.gamma_buf[self._next_n:stop_n],
                                        self.comp_xi_buf[self._next_n:stop_n], self.xi_buf[self._next_n:stop_n],
                                        ignore_zeros=self.ignore_zeros, epsabs=self.epsabs, epsrel=self.epsrel)
        self._set_next_n(stop_n)
