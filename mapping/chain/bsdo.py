import orthpol as orthpol
from mapping.utils.convergence import check_array_convergence
import numpy as np


def get_bsdo(J, domain, nof_coefficients, orthpol_ncap=None):
    """
        Computes chain bsdo coefficients (TEDOPA coefficients, see Chin et al., J. Math. Phys. 51, 092109 (2010))
        using monic orthogonal polynomial recurrence
    :param J: Spectral density of the bath
    :param domain: Domain (support) of the spectral density
    :param nof_coefficients: Number of coefficients to be calculated
    :param orthpol_ncap: Accuracy parameter for orthpol (internal discretization parameter). If set None a default of
                         60000 is used.
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings)
    """
    if orthpol_ncap is None:
        orthpol_ncap = 60000
    poly = orthpol.OrthogonalPolynomial(nof_coefficients - 1, left=domain[0], right=domain[1], wf=J, ncap=orthpol_ncap)
    alpha, beta = np.array(poly.alpha), np.array(poly.beta)
    return np.sqrt(beta[0]), alpha, np.sqrt(beta[1:])


def get_bsdo_from_convergence(J, domain, nof_coefficients, min_ncap=10000, step_ncap=1000,
                              max_ncap=60000, stop_abs=1e-10, stop_rel=1e-10, threshold=1e-15):
    """
        Computes. Determines the accuracy parameter orthpol_ncap for the
        computation using a convergence condition for the chain coefficients omega, t. Iteratively
        increases that accuracy every step.
        Convergence condition considers all bath energies and couplings in every step
    :param J: Spectral density of the bath as vectorized python function of one argument
    :param domain: Domain (support) of the spectral density as list/tuple/numpy array like [a, b]
    :param nof_coefficients: Number of chain coefficients to be calculated
    :param min_ncap: Minimum accuracy to use for the computaton (start of the convergence check)
                     Must be > 0
    :param max_ncap: Maximum accuracy parameter. Forces exit if ncap reaches that value without convergence
                     Must be > min_ncap
    :param step_ncap: Number of star coefficients to be added in each step of the convergence
    :param stop_rel: Target relative deviation between successive steps. May be None if stop_abs is not None
    :param stop_abs: Target absolute deviation between successive steps. May be None if stop_rel is not None
    :param threshold: Threshold, for which smaller numbers are treated as numerically 0 for the purposes
                      of the convergence condition
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings)
    """
    ncap = find_bsdo_ncap(J, domain, nof_coefficients, min_ncap=min_ncap, step_ncap=step_ncap,
                          max_ncap=max_ncap, stop_abs=stop_abs, stop_rel=stop_rel, threshold=threshold)
    info = dict()
    info['ncap'] = ncap
    c0, omega, t = get_bsdo(J, domain, nof_coefficients, orthpol_ncap=ncap)
    return c0, omega, t, info


def find_bsdo_ncap(J, domain, nof_coefficients, min_ncap=10000, step_ncap=1000,
                   max_ncap=60000, stop_abs=1e-10, stop_rel=1e-10, threshold=1e-15):
    """
        See get_bsdo_from_convergence method
    """
    assert 0 < min_ncap < max_ncap
    assert step_ncap > 0
    poly = orthpol.OrthogonalPolynomial(nof_coefficients - 1, left=domain[0], right=domain[1], wf=J, ncap=min_ncap)
    last_alpha, last_beta = np.array(poly.alpha), np.array(poly.beta)
    for ncap in range(min_ncap+step_ncap, max_ncap+step_ncap, step_ncap):
        poly = orthpol.OrthogonalPolynomial(nof_coefficients - 1, left=domain[0], right=domain[1], wf=J, ncap=ncap)
        curr_alpha, curr_beta = np.array(poly.alpha), np.array(poly.beta)
        if check_array_convergence(last_alpha, last_beta, curr_alpha, curr_beta, stop_abs, stop_rel,
                                   threshold=threshold):
            return ncap
        last_alpha, last_beta = last_alpha, last_beta
    print('Did not reach convergence, use maximum ncap = ' + str(max_ncap))
    return max_ncap
