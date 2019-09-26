import numpy as np
import matplotlib.pyplot as plt
import mapping


# Ohmic spectral density parameters:
alpha = 1
s = 1
omega_c = 1

nof_coefficients = 50

# Discretization and chain mapping type:
disc_type = 'gk_quad'
mapping_type = 'sp_hes'

# Discretization accuracy parameter:
ncap = 100

# Domain of the spectral density. Must choose manual cutoff value here
domain = [0.0, 25]

J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-(x/omega_c))

c0, omega, t, info = \
    mapping.chain.get(J, domain, nof_coefficients, ncap=ncap, disc_type=disc_type, mapping_type=mapping_type,
                      interval_type='lin')

print('Residual from the tridiagonalization: ', info['res'])

print('System to bath coupling: ', c0)


plt.figure(1)
plt.plot(omega, '.')
plt.title('Bath energies')
plt.xlabel('Site index')
plt.ylabel('$\\omega$')
plt.figure(2)
plt.plot(t, '.')
plt.title('Bath couplings')
plt.xlabel('Bond index')
plt.ylabel('$t$')
plt.show()