"""
    Discretized bath for the generation of BSDO type coefficients using monic orthogonal polynomial recurrence
"""
import orthpol as orthpol
import numpy as np
from mapping.star.discretized_bath.base.asymmetric import BaseDiscretizedAsymmetricBath
from mapping.star.discretized_bath.stopcoeff import StopCoefficients
from mapping.utils.convert import convert_chain_to_star


class BSDODiscretizedAsymmetricBath(BaseDiscretizedAsymmetricBath):
    def __init__(self, J, domain, max_nof_coefficients=100, **kwargs):
        """
            Generates BSDO coefficients from a spectral density J, by diagonalizing the tridiagonal matrix
            of polynomial recurrence coefficients
        :param J: Spectral density. A function defined on 'domain', must be >0 in the inner part of domain
        :param domain: List/tuple of two elements for the left and right boundary of the domain of J
        :param max_nof_coefficients: Size of the buffers which hold gamma and xi coefficients (maximum number of
                         these coefficients that can be calculated)
        :param kwargs: may contain 'orthpol_ncap' to change the internal discretization of orthpol, default is 60000
                                   'mp_dps' to change the number o decimals with which the diagonalization is performed
                                            defult is 30
                                   'force_sp' forces scipy eigh for the diagonalization, default is False
        """
        self.J = J
        try:
            orthpol_ncap = kwargs['orthpol_ncap']
        except KeyError:
            orthpol_ncap = 60000
        try:
            mp_dps = kwargs['mp_dps']
        except KeyError:
            mp_dps = 30
        try:
            force_sp = kwargs['force_sp']
        except KeyError:
            force_sp = False
        try:
            get_trafo = kwargs['get_trafo']
        except KeyError:
            get_trafo = False
        super().__init__(self.compute_coefficients, max_nof_coefficients=max_nof_coefficients)
        # Preallocate arrays for the coefficients
        alpha, beta = self.get_monic_recurrence(max_nof_coefficients, domain, J, orthpol_ncap)
        self._eta_0 = beta[0]
        self.gamma_buf[:], self.xi_buf[:], info = \
            convert_chain_to_star(np.sqrt(beta[0]), alpha, np.sqrt(beta[1::]), force_sp=force_sp, mp_dps=mp_dps,
                                  sort_by=None, get_trafo=get_trafo)
        self.info = info
        self._set_next_n(max_nof_coefficients)

    @property
    def eta_0(self):
        """
            Returns sum_i gamma_i (of the currently calculated coefficients)
        """
        return self._eta_0

    def compute_coefficients(self, stop_n):
        """
            Immediately raises a StopCoefficients exception, because everything is already calculated in the constructor
        """
        raise StopCoefficients

    @staticmethod
    def get_monic_recurrence(nof_coefficients, support, wf, ncap):
        """
            Calculates two-point coefficients(alpha, beta) for monic polynomials using py-orthpol
        :param nof_coefficients: Number of coefficients to calculate
        :param support: Support of the orthognal polynomials
        :param wf: Weight function for the inner product
        :param ncap: Used by py-orthpol to determine the accuracy for the calculation of the coefficients coefficients.
                     Must be <= 60000. A larger ncap results in greater accuracy at the cost of runtime.
        :return: Tuple of lists (alpha, beta) of length nof_coefficients each
        """
        # First positional argument of OrthogonalPolynomial is actually the order of the polynomial to generate
        # OrthogonalPolynomial actually generates one additional coefficient (if order 2 was selected, it would
        # generate coefficients alpha_0, alpha_1 and alpha_2 for example!)
        poly = orthpol.OrthogonalPolynomial(nof_coefficients-1, left=support[0], right=support[1], wf=wf, ncap=ncap)
        return np.array(poly.alpha), np.array(poly.beta)
