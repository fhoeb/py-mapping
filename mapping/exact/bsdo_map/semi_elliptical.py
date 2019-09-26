import numpy as np


def get_semi_elliptical_coefficients(alpha, nof_coefficients=100):
    """
        Generates exact bsdo chain coefficients for a bath with semi-elliptical spectral density:
        Delta(eps) = alpha*(1-x^2)^(1/2) , eps in [-1.0, 1.0]
    :param alpha: Prefactor
    :param nof_coefficients: Number of coefficients to generate
    :return:
    """
    c0 = np.sqrt(np.pi/2 * alpha)
    omega = np.zeros(nof_coefficients)
    t = np.empty(nof_coefficients-1)
    t.fill(1/2)
    return c0, omega, t
