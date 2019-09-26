import mapping
import pytest
import numpy as np


@pytest.mark.slow
@pytest.mark.parametrize('alpha, s, omega_c', [(1, 1, 1), (1, 2, 0.5), (1, 0.5, 2)])
@pytest.mark.parametrize('nof_coefficients', [20, 30])
@pytest.mark.parametrize('mapping_type', ['lan_bath', 'hou', 'sp_hes'])
def test_mappings(alpha, s, omega_c, nof_coefficients, mapping_type):
    """
        Test different mapping types using a cut ohmic spectral density
    """
    # We compare with exact bsdo coefficients here, so accuracy won't be that high
    acc = 1e-3
    ncap = 1000
    domain = [0.0, omega_c]
    J = lambda x: alpha * omega_c * (x / omega_c) ** s
    c0, omega, t, info = \
        mapping.chain.get(J, domain, nof_coefficients, ncap=ncap, disc_type='sp_quad', mapping_type=mapping_type,
                          residual=True, interval_type='lin')
    c0_ref, omega_ref, t_ref = mapping.exact.get_ohmic_cutoff_coefficients(alpha, s, omega_c, nof_coefficients)
    assert np.abs(c0-c0_ref)/np.abs(c0_ref) < acc
    assert np.max(np.abs(omega - omega_ref) / np.abs(omega_ref)) < acc
    assert np.max(np.abs(t - t_ref) / np.abs(t_ref)) < acc


@pytest.mark.slow
@pytest.mark.parametrize('Delta_0, r', [(1, 1), (1, 2), (1, 0.5)])
@pytest.mark.parametrize('Lambda', [1.1, 1.5])
@pytest.mark.parametrize('nof_coefficients', [20, 30])
@pytest.mark.parametrize('mapping_type', ['hou', 'sp_hes'])
@pytest.mark.parametrize('stepwise', [True, False])
def test_mapping_convergence(Delta_0, r, Lambda, nof_coefficients, mapping_type, stepwise):
    """
        Test different mapping types for their convergence using a soft-gap model hybridization function and
        logarithmic discretization
    """
    acc = 1e-7
    min_ncap = 50
    max_ncap = 1000
    domain = [-1.0, 1.0]
    J = lambda x: Delta_0 * np.abs(x) ** r
    if stepwise:
        c0, omega, t, info = \
            mapping.chain.get_from_stepwise_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                                                        interval_type='log', mapping_type=mapping_type,
                                                        permute=None, residual=True, ignore_zeros=True,
                                                        Lambda=Lambda, symmetric=True,
                                                        max_ncap=max_ncap, min_ncap=min_ncap,
                                                        stop_rel=1e-8, stop_abs=1e-8)
    else:
        c0, omega, t, info = \
            mapping.chain.get_from_convergence(J, domain, nof_coefficients,disc_type='sp_quad',
                                               interval_type='log', mapping_type=mapping_type,
                                               permute=None, residual=True, ignore_zeros=True,
                                               Lambda=Lambda, symmetric=True,
                                               max_ncap=max_ncap, min_ncap=min_ncap,
                                               stop_rel=1e-8, stop_abs=1e-8)
    c0_ref, omega_ref, t_ref = mapping.exact.get_nrg_soft_gap_coefficients(Delta_0, r, Lambda, nof_coefficients)
    assert np.abs(c0-c0_ref)/np.abs(c0_ref) < acc
    # exact omega is 0, use abs-error here
    assert np.max(np.abs(omega - omega_ref)) < acc
    assert np.max(np.abs(t - t_ref) / np.abs(t_ref)) < acc

