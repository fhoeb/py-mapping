import numpy as np


def get_const_hyb_coefficients(V, nof_coefficients=100):
    """
        Generates exact bsdo chain coefficients for a bath with constant spectral density:
        Delta(eps) = Delta_0 = 1/2 * V^2 , eps in [-1.0, 1.0]
    :param V: Coupling strength
    :param nof_coefficients: Number of coefficients to generate
    :return:
    """
    c0 = V
    omega = np.zeros(nof_coefficients)
    n = np.arange(0, nof_coefficients-1)
    t = np.sqrt((n + 1) ** 2 / (4 * (n + 1) ** 2 - 1))
    return c0, omega, t
