import numpy as np


def get_soft_gap_coefficients(Delta_0, r, nof_coefficients=100):
    """
        Coefficients for the exact bsdo chain mapping for the soft gap model spectral density:
        Delta(eps) = Delta_0 * |eps|^r , eps in [-1.0, 1.0]
        (Calculated using the limit of Lambda -> 0 for the corresponding NRG coefficients which were taken from:
        R Bulla et al 1997 J. Phys.: Condens. Matter 9 10463–10474; PII: S0953-8984(97)85805-5)
    :param Delta_0: Coupling strength
    :param r: Soft gap exponent
    :param nof_coefficients: Number of coefficients to generate
    :return:
    """
    eta_0 = 2 * Delta_0 * 1/(r + 1)
    c0 = np.sqrt(eta_0)
    omega = np.zeros(nof_coefficients)
    t = np.empty(nof_coefficients-1)
    for n in range(nof_coefficients-1):
        if n % 2 == 0:
            t[n] = np.sqrt((1+n+r)**2 / (4*(n+1)**2 - 1 + 4*r + r**2 + 4*n*r))
        else:
            t[n] = np.sqrt((1+n)**2 / (4*(n+1)**2 - 1 + 4*r + r**2 + 4*n*r))
    return c0, omega, t
