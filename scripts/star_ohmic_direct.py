import numpy as np
import matplotlib.pyplot as plt
import mapping


# Ohmic spectral density parameters:
alpha = 1
s = 1
omega_c = 1

nof_coefficients = 50

# Discretization and chain mapping type:
disc_type = 'sp_quad'


# Domain of the spectral density. Must choose manual cutoff value here
domain = [0.0, 25]

J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-(x/omega_c))

gamma, xi = \
    mapping.star.get(J, domain, nof_coefficients,disc_type=disc_type, interval_type='lin')


plt.figure(1)
plt.plot(xi, '.')
plt.title('Bath energies')
plt.xlabel('Bath site')
plt.ylabel('$\\omega$')
plt.figure(2)
plt.plot(gamma, '.')
plt.title('System to bath couplings')
plt.xlabel('Bath site')
plt.ylabel('$t$')
plt.show()