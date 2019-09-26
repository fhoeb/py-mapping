import numpy as np
from scipy.special import gamma


def get_ohmic_coefficients(alpha, s, omega_c, nof_coefficients=100):
    """
        Generates exact bsdo chain coefficients for a bosonic bath for ohmic spectral density with hard cutoff:
        J(w) = alpha * omega_c * (w / omega_c) ** s * exp(-w/omega_c), w in [0, inf]
    :param alpha: Coupling strength
    :param s: Ohmic exponent
    :param omega_c: Cutoff
    :param nof_coefficients: Number of coefficients to generate
    :return:
    """
    eta_0 = alpha*omega_c**2 * gamma(s+1)
    c0 = np.sqrt(eta_0)
    n = np.arange(0, nof_coefficients)
    omega = omega_c * (2*n + 1 + s)
    n = n[:-1]
    t = omega_c * np.sqrt((n + 1)*(n + s + 1))
    return c0, omega, t