import numpy as np
from mapping.utils.gauss_kronrod_roots import GK_roots, orders
from mapping.utils.gauss_kronrod_weights import GK_weights


def check_converged(gauss_gamma, kronrod_gamma, gauss_xi, kronrod_xi, epsrel, epsabs):
    """
        Checks for convergence between Gauss and Kronrod integrals
    """
    # Check for gamma:
    # only need to check nonzero part, as zeros are generated only if 'ignore_zeros' is set True anyway
    nonzero = np.nonzero(gauss_gamma)
    absdiff = np.abs(gauss_gamma[nonzero] - kronrod_gamma[nonzero])
    if not (np.all(absdiff < epsabs) and np.all(absdiff / np.abs(kronrod_gamma[nonzero]) < epsrel)):
        return False
    absdiff = np.abs(gauss_xi[nonzero] - kronrod_xi[nonzero])
    if not (np.all(absdiff < epsabs) and np.all(absdiff / np.abs(kronrod_xi[nonzero]) < epsrel)):
        return False
    return True


def _compute_quadrature_integrals(J, mid, dx, gamma_buf, xi_buf, roots, weights, ignore_zeros=False):
    """
        Implements a vectorized quadrature rule integration for the discretization of a spectral density J
        Computes:
        gamma_i = sqrt(int_i^i+1 J(x) dx)
        xi_i = int_i^i+1 J(x) * x dx/ gamma_i^2
        where i specifies an interval in the domain of J.
        Fills the xi and gamma buffers with the computed coefficients.
    :param J: Spectral density, vectorized to accept a numpy array as input
    :param mid: Midpoints of the intervals over which to integrate
    :param dx: Widths of the intervals over which to integrate
    :param gamma_buf: Buffer for the calculated gamma_i
    :param xi_buf: Buffer for the calculated xi_i
    :param roots: Roots for the Gauss or Kronrod Quadrature rule
    :param weights: Weights for the Gauss or Kronrod Quadrature rule
    :param ignore_zeros: If set True,  any gamma_i, that are numerically zero during the computation will not be used
                         for the computation of xi_i. Instead the corresponding xi_i will also be set to 0 (which
                         is physically reasonable, but mathematically suspicious)
    """

    def Jx(x): return x*J(x)
    dx = dx/2
    np.multiply(weights[0], J(mid + dx * roots[0]), out=gamma_buf)
    np.multiply(weights[0], Jx(mid + dx * roots[0]), out=xi_buf)
    for root, weight in zip(roots[1::], weights[1::]):
        gamma_buf += weight * J(mid + dx * root)
        xi_buf += weight * Jx(mid + dx * root)
    if not ignore_zeros:
        xi_buf /= gamma_buf
    else:
        gamma_nonzero = np.nonzero(gamma_buf)
        gamma_zeros = np.nonzero(gamma_buf == 0)
        xi_buf[gamma_nonzero] /= gamma_buf[gamma_nonzero]
        xi_buf[gamma_zeros] = 0
    np.multiply(dx, gamma_buf, out=gamma_buf)
    np.sqrt(gamma_buf, out=gamma_buf)


def compute_gk_quadrature_integrals(J, mid, dx, comp_gamma_buf, gamma_buf, comp_xi_buf, xi_buf, ignore_zeros=False,
                                    epsabs=1e-11, epsrel=1e-11):
    """
        Implements a vectorized Gauss-Kronrod quadrature rule integration for the discretization of a spectral density J
        Computes:
        gamma_i = sqrt(int_i^i+1 J(x) dx)
        xi_i = int_i^i+1 J(x) * x dx/ gamma_i^2
        where i specifies an interval in the domain of J.
        Fills the xi and gamma buffers with the computed coefficients.
    :param J: Spectral density, vectorized to accept a numpy array as input
    :param mid: Midpoints of the intervals over which to integrate
    :param dx: Widths of the intervals over which to integrate of the same size as mid
    :param gamma_buf: Buffer for the calculated gamma_i (must be as large as mid and dx)
    :param comp_gamma_buf: Buffer for the comparison between Gauss and Kronrod coefficients
                          (must be as large as mid and dx)
    :param xi_buf: Buffer for the calculated xi_i (must be as large as mid and dx)
    :param comp_xi_buf: Buffer for the comparison between Gauss and Kronrod coefficients
                        (must be as large as mid and dx)
    :param ignore_zeros: If set True,  any gamma_i, that are numerically zero during the computation will not be used
                         for the computation of xi_i. Instead the corresponding xi_i will also be set to 0 (which
                         is physically reasonable, but mathematically suspicious)
    :param epsabs: Convergence condition on the absolute difference between Gauss and Kronrod integration results
    :param epsrel: Convergence condition on the relative difference between Gauss and Kronrod integration results
    :return:
    """
    for order in orders:
        roots = GK_roots[order]
        weights = GK_weights[order]
        _compute_quadrature_integrals(J, mid, dx, comp_gamma_buf, comp_xi_buf, roots[0], weights[0],
                                      ignore_zeros=ignore_zeros)
        _compute_quadrature_integrals(J, mid, dx, gamma_buf, xi_buf, roots[1], weights[1], ignore_zeros=ignore_zeros)
        if check_converged(comp_gamma_buf, gamma_buf, comp_xi_buf, xi_buf, epsrel, epsabs):
            return
    print('Target accuracy could not be reached with all coefficients')
