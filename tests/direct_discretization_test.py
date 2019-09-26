import mapping
import pytest
import numpy as np


@pytest.mark.fast
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5)])
@pytest.mark.parametrize('nof_coefficients', [100, 1000, 10000])
@pytest.mark.parametrize('disc_type', ['quad', 'gk_quad'])
def test_direct_discretization_quad(alpha, s, omega_c, nof_coefficients, disc_type):
    """
        Test different direct discretization types (quad, gk_quad, sp_quad), that they produce the
        same output star coefficients
    """
    # We compare with bsdo coefficients here, so accuracy won't be that high
    acc = 1e-12
    domain = [0.0, omega_c]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s
    gamma, xi = \
        mapping.star.get(J, domain, nof_coefficients, disc_type=disc_type, interval_type='lin',
                         sort_by='xi_a')
    gamma_ref, xi_ref = \
        mapping.star.get(J, domain, nof_coefficients, disc_type='sp_quad', interval_type='lin',
                         sort_by='xi_a')
    assert np.max(np.abs(gamma - gamma_ref)/np.abs(gamma_ref)) < acc
    assert np.max(np.abs(xi - xi_ref) / np.abs(xi_ref)) < acc


@pytest.mark.fast
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5)])
@pytest.mark.parametrize('nof_coefficients', [20, 30])
@pytest.mark.parametrize('disc_type, ncap', [('quad', 1000000), ('gk_quad', 1000000), ('trapz', 1000000),
                                             ('midpt', 1000000)])
def test_direct_discretization_convergence(alpha, s, omega_c, nof_coefficients, disc_type, ncap):
    """
        Test that different (vectorized) direct discretization types do converge towards the bsdo coefficients for
        a high enough discretization ncap
    """
    # We compare with bsdo coefficients here, so accuracy won't be that high
    acc = 1e-9
    domain = [0.0, omega_c]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s
    c0, omega, t, info = \
        mapping.chain.get(J, domain, nof_coefficients, ncap=ncap, disc_type=disc_type, mapping_type='lan_bath',
                          residual=True, interval_type='lin')
    c0_ref, omega_ref, t_ref = mapping.exact.get_ohmic_cutoff_coefficients(alpha, s, omega_c, nof_coefficients)
    assert np.abs(c0-c0_ref)/np.abs(c0) < acc
    assert np.max(np.abs(omega - omega_ref) / np.abs(omega)) < acc
    assert np.max(np.abs(t - t_ref) / np.abs(t)) < acc