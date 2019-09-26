import numpy as np
from mapping.utils.gauss_legendre_roots import roots
from mapping.utils.gauss_legendre_weights import weights
max_nof_pts = min(len(roots), len(weights)) + 1


def compute_quadrature_integrals(J, mid, dx, gamma_buf, xi_buf, ignore_zeros=False, order=10):
    """
        Implements a fixed order vectorized Gauss-Legendre quadrature rule integration for the discretization of a
        spectral density J. Computes:
        gamma_i = sqrt(int_i^i+1 J(x) dx)
        xi_i = int_i^i+1 J(x) * x dx/ gamma_i^2
        where i specifies an interval in the domain of J.
        Fills the xi and gamma buffers with the computed coefficients.
    :param J: Spectral density, vectorized to accept a numpy array as input
    :param mid: Midpoints of the intervals over which to integrate
    :param dx: Widths of the intervals over which to integrate of the same size as mid
    :param gamma_buf: Buffer for the calculated gamma_i (must be as large as mid and dx)
    :param xi_buf: Buffer for the calculated xi_i (must be as large as mid and dx)
    :param ignore_zeros: If set True,  any gamma_i, that are numerically zero during the computation will not be used
                         for the computation of xi_i. Instead the corresponding xi_i will also be set to 0 (which
                         is physically reasonable, but mathematically suspicious)
    :param order: Order of the Quadrature rule (allowed are al values between n=2 and n=64, including those two)
    :return:
    """
    assert order <= max_nof_pts

    def Jx(x): return x*J(x)
    leg_roots = roots[order-2]
    leg_weights = weights[order-2]
    dx = dx/2
    np.multiply(leg_weights[0], J(mid + dx * leg_roots[0]), out=gamma_buf)
    np.multiply(leg_weights[0], Jx(mid + dx * leg_roots[0]), out=xi_buf)
    for root, weight in zip(leg_roots[1::], leg_weights[1::]):
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
