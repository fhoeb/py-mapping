import numpy as np
import mapping


# Ohmic spectral density parameters:
alpha = 1
s = 1
omega_c = 1

nof_coefficients = 5

# Discretization and chain mapping type:
disc_type = 'gk_quad'
mapping_type = 'lan_bath'

# Discretization accuracy parameter:
ncap = 1000000

# Domain of the spectral density. Must choose manual cutoff value here
domain = [0.0, 25]

J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-(x/omega_c))

c0_ref, omega_ref, t_ref, info = \
    mapping.chain.get(J, domain, nof_coefficients, ncap=ncap, disc_type=disc_type, mapping_type=mapping_type,
                      interval_type='lin')

print('Residual from the tridiagonalization: ', info['res'])

# Map to the star coefficients
gamma, xi, info = \
    mapping.convert_chain_to_star(c0_ref, omega_ref, t_ref, get_trafo=True)


# Map back to the chain coefficients
c0, omega, t, info = \
    mapping.convert_star_to_chain_lan(gamma, xi, get_trafo=True)


print('Residuals of the back and forth mapping: ')
print('Of the system-bath coupling: ', np.abs((c0 - c0_ref) / c0_ref))
print('Of the bath energies: ', np.max(np.abs((omega - omega_ref)/omega_ref)))
print('Of the bath-bath couplings: ', np.max(np.abs((t - t_ref)/t_ref)))