import mapping
import pytest
import numpy as np


@pytest.mark.fast
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5), (1, 0.5, 2)])
@pytest.mark.parametrize('nof_coefficients', [20, 30, 50])
def test_conversion(alpha, s, omega_c, nof_coefficients):
    """
        Test for the regular star-chain and chain-star conversion
    """
    acc = 1e-12
    domain = [0.0, np.inf]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-x/omega_c)
    c0, omega, t, info = mapping.chain.get(J, domain, nof_coefficients, disc_type='bsdo', orthpol_ncap=10000)
    gamma, xi, info = mapping.convert_chain_to_star(c0, omega, t)
    c0_conv, omega_conv, t_conv, info = mapping.convert_star_to_chain(gamma, xi, residual=True)
    assert np.abs(c0-c0_conv)/np.abs(c0) < acc
    assert np.max(np.abs(omega - omega_conv) / np.abs(omega)) < acc
    assert np.max(np.abs(t - t_conv) / np.abs(t)) < acc


@pytest.mark.fast
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5), (1, 0.5, 2)])
@pytest.mark.parametrize('nof_coefficients', [20, 30, 50])
def test_conversion_lanczos(alpha, s, omega_c, nof_coefficients):
    """
        Test for the star-chain and chain-star conversion, using lanczos for the star-chain one
    """
    acc = 1e-12
    domain = [0.0, np.inf]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-x/omega_c)
    c0, omega, t, info = mapping.chain.get(J, domain, nof_coefficients, disc_type='bsdo', orthpol_ncap=10000)
    gamma, xi, info = mapping.convert_chain_to_star(c0, omega, t)
    c0_conv, omega_conv, t_conv, info = mapping.convert_star_to_chain_lan(gamma, xi, residual=True)
    assert np.abs(c0 - c0_conv)/np.abs(c0) < acc
    assert np.max(np.abs(omega - omega_conv) / np.abs(omega)) < acc
    assert np.max(np.abs(t - t_conv) / np.abs(t)) < acc