"""
    Discretized bath for the generation of direct asymmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using scipy quad
"""
import numpy as np
from scipy import integrate
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.utils.integration_defaults import default_epsabs, default_epsrel, default_limit
from mapping.star.discretized_bath.stopcoeff import StopCoefficients
from mapping.star.discretized_bath.intervals import get_asymmetric_interval_points


class SpQuadDiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
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
                                   'epsabs': absolute tolerance for the scipy integrations, default is 1e-11
                                   'epsrel': relative tolerance for the scipy integrations, default is 1e-11
                                   'limit': limit parameter for the scipy quad function, default is 100
        """
        assert not np.isinf(domain[1])
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
        try:
            self.limit = kwargs['limit']
        except KeyError:
            self.limit = default_limit
        self.x_pts = get_asymmetric_interval_points(domain, max_nof_coefficients, interval_type=interval_type,
                                                    get_spacing=False, **kwargs)
        self.J = J
        self.interval_type = interval_type
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)

    def compute_coefficients(self, stop_n):
        """
            Calculates the discretization coefficients up to stop_n
        :param stop_n: Index up to which new coefficients are calculated
        """
        x_pts = self.x_pts[::-1] if self.interval_type == 'log' else self.x_pts
        for n in range(self._next_n, stop_n):
            try:
                a, b = x_pts[n], x_pts[n + 1]
                # Must invert the view, because for log-discretization the positive domain grid points are in
                # inverted order
                if self.interval_type == 'log':
                    a, b = b, a
            except IndexError:
                raise StopCoefficients
            gamma_sq, err = \
                integrate.quad(self.J, a, b, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)
            xi_numerator, err = \
                integrate.quad(lambda x: x * self.J(x), a, b, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)
            self.gamma_buf[n] = np.sqrt(gamma_sq)
            if self.ignore_zeros and gamma_sq == 0:
                self.xi_buf[n] = 0
            else:
                self.xi_buf[n] = xi_numerator/gamma_sq
            self._update_next_n(1)
