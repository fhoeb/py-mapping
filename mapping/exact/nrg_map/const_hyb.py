import numpy as np


def get_nrg_const_hyb_coefficients(V, Lambda, nof_coefficients, corrected=True):
    """
        Coefficients for the exact chain mapping for a constant hybridization in the SIAM, V_k = V
        (equivalent to setting Delta(eps) = pi*V^2 / 2, for eps in [-1.0, 1.0])
        using logarithmic discratization
        Source:
        R Bulla et al 2008 Rev. Mod. Phys. 80; DOI: 10.1103/RevModPhys.80.395
    :param V: coupling strength
    :param Lambda: Logarithmic discretization (>1.0)
    :param nof_coefficients: number of coefficients (number of bath site energies omega)
    :param corrected: If a heuristic 'correction factor' should be applied to the system-bath coupling c0
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings)
    """
    assert Lambda > 1.0
    if corrected:
        c0 = V * np.log(Lambda) * (Lambda + 1)/(2*(Lambda - 1))
    else:
        c0 = V
    omega = np.zeros(nof_coefficients)
    n = np.arange(0, nof_coefficients-1)
    t = (1 + Lambda ** (-1)) * (1 - Lambda ** (-n - 1)) * Lambda ** (-n / 2) / \
        (2 * np.sqrt(1 - Lambda ** (-2 * n - 1)) * np.sqrt(1 - Lambda ** (-2 * n - 3)))
    return c0, omega, t