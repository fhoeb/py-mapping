import numpy as np


def get_ohmic_cutoff_coefficients(alpha, s, omega_c, nof_coefficients=100):
    """
        Generates exact bsdo chain coefficients for a bosonic bath for ohmic spectral density with hard cutoff:
        J(w) = alpha * omega_c * (w / omega_c) ** s, w in [0, omega_c]
    :param alpha: Coupling strength
    :param s: Ohmic exponent
    :param omega_c: Cutoff
    :param nof_coefficients: Number of coefficients to generate
    :return:
    """
    eta_0 = alpha*omega_c**2 * 1/(s+1)
    c0 = np.sqrt(eta_0)
    n = np.arange(0, nof_coefficients)
    omega = omega_c/2 * (1 + (s**2)/((s + 2*n)*(2 + s + 2*n)))
    n = n[:-1]
    t = omega_c * (1 + n)*(1 + s + n)/((s + 2 + 2*n)*(3 + s + 2*n)) * np.sqrt((3 + s + 2*n)/(1 + s + 2*n))
    return c0, omega, t
