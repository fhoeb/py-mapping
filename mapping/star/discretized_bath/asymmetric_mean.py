"""
    Discretized bath for the generation of direct asymmetric discretization coefficients, where the integrals for
    the couplings and energies are evaluated using a heuristic method called mean discretization.
    Introduced in: de Vega et al.,  Phys. Rev. B 92, 155126 (2015)
"""
import numpy as np
from scipy.integrate import quad
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.utils.integration_defaults import default_epsabs, default_epsrel, default_limit
from mapping.star.discretized_bath.stopcoeff import StopCoefficients


class MeanDiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
    def __init__(self, J, domain, max_nof_coefficients=100, **kwargs):
        """
            Generates direct discretization coefficients from a spectral density J, by
            mean discretization (see de Vega et al.,  Phys. Rev. B 92, 155126 (2015) for details on the method)
            Computes max_nof_coefficients coefficients directly!
        :param J: Spectral density. A function defined on 'domain', must be >0 in the inner part of domain
        :param domain: List/tuple of two elements for the left and right boundary of the domain of J
        :param max_nof_coefficients: Size of the buffers which hold gamma and xi coefficients (maximum number of
                                     these coefficients that can be calculated)
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
        self.J = J
        self.Jx = lambda x: J(x) * x
        self.domain = domain
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)
        try:
            self.gamma_buf[:], self.xi_buf[:] = self.get_mean_coefficients(max_nof_coefficients)
        except ZeroDivisionError:
            print('Cannot calculate ' + str(max_nof_coefficients) + ' coefficients. Encountered div/0')
            raise
        self._set_next_n(max_nof_coefficients)

    def get_interval_avg(self, a, b):
        """
            Returns the average of J in the interval [1a, b]
        """
        return quad(self.Jx, a=a, b=b, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)[0] / \
               quad(self.J, a=a, b=b, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)[0]

    def get_mean_coefficients(self, nof_coefficients):
        """
           Calculates the mean discretization coefficients
        """
        interval_points = np.empty(nof_coefficients+2)
        # include the endpoints of the interval
        interval_points[0] = self.domain[0]
        interval_points[-1] = self.domain[1]
        x0 = self.get_interval_avg(self.domain[0], self.domain[1])
        interval_points[1] = x0
        # iteratively determine the points, that divide J by equal weight
        for n in range(2, nof_coefficients+1):
            last_points = np.empty(n+1)
            last_points[0] = self.domain[0]
            last_points[-1] = self.domain[1]
            last_points[1:-1] = interval_points[1:n]
            for pt_idx in range(n):
                interval_points[pt_idx+1] = self.get_interval_avg(last_points[pt_idx], last_points[pt_idx+1])
        # Calculate the couplings in the above determined intervals
        couplings = np.empty(nof_coefficients)
        for pt_idx in range(1, nof_coefficients+1):
            a = (interval_points[pt_idx-1] + interval_points[pt_idx])/2 if pt_idx > 1 else interval_points[0]
            b = (interval_points[pt_idx] + interval_points[pt_idx+1])/2 if pt_idx < nof_coefficients else \
                interval_points[nof_coefficients+1]
            couplings[pt_idx-1] = np.sqrt(quad(self.J, a=a, b=b, epsabs=self.epsabs, epsrel=self.epsrel,
                                               limit=self.limit)[0])
        return couplings, interval_points[1:-1]

    def compute_coefficients(self, stop_n):
        """
            Immediately raises a StopCoefficients exception, because everything is already calculated in the constructor
        """
        raise StopCoefficients
