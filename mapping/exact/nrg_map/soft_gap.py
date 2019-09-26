import numpy as np


def get_nrg_soft_gap_coefficients(Delta_0, r, Lambda, nof_coefficients):
    """
        Coefficients for the exact chain mapping for the soft gap model:
        Delta(eps) = Delta_0 * |eps|^r , eps in [-1.0, 1.0]
        using logarithmic discretization
        Source:
        R Bulla et al 1997 J. Phys.: Condens. Matter 9 10463–10474; PII: S0953-8984(97)85805-5
    :param Delta_0: coupling strength
    :param r: soft gap exponent
    :param Lambda: Logarithmic discretization parameter (>1.0)
    :param nof_coefficients: Number of coefficients to be calculated (number of bath site energies omega)
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings),
    """
    assert Lambda > 1.0
    eta_0 = 2 * Delta_0 * (1/(r+1))
    omega = np.zeros(nof_coefficients)
    t = np.empty(nof_coefficients-1)
    for n in range(nof_coefficients-1):
        if n % 2 == 0:
            t[n] = Lambda**(-n/2) * (r+1)/(r+2) * (1 - Lambda**(-(r+2)))/(1 - Lambda**(-(r+1))) * \
                   (1 - Lambda**(-(n+r+1)))/np.sqrt((1 - Lambda**(-(2*n + r + 1))) * (1 - Lambda**(-(2*n + r + 3))))
        else:
            t[n] = Lambda**(-(n+r)/2) * (r+1)/(r+2) * (1 - Lambda**(-(r+2)))/(1 - Lambda**(-(r+1))) * \
                   (1 - Lambda**(-(n+1)))/np.sqrt((1 - Lambda**(-(2*n + r + 1))) * (1 - Lambda**(-(2*n + r + 3))))
    return np.sqrt(eta_0), omega, t