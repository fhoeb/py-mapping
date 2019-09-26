import mapping
import pytest
import numpy as np


@pytest.mark.fast
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5), (1, 0.5, 2)])
@pytest.mark.parametrize('nof_coefficients', [20, 30, 50])
def test_ohmic_bsdo(alpha, s, omega_c, nof_coefficients):
    """
        Test for bsdo coefficient generation of ohmic coefficients
    """
    acc = 1e-12
    domain = [0.0, np.inf]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-x/omega_c)
    c0, omega, t, info = mapping.chain.get(J, domain, nof_coefficients, disc_type='bsdo', orthpol_ncap=10000)
    c0_ref, omega_ref, t_ref = mapping.exact.get_ohmic_coefficients(alpha, s, omega_c, nof_coefficients)
    c0_rel = np.abs((c0 - c0_ref) / c0_ref)
    omega_rel = np.abs((omega - omega_ref)/omega_ref)
    t_rel = np.abs((t - t_ref)/t_ref)
    assert c0_rel < acc
    assert np.max(omega_rel) < acc
    assert np.max(t_rel) < acc