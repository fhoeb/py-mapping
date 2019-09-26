import numpy as np
import matplotlib.pyplot as plt
import mapping


# Ohmic spectral density parameters:
alpha = 1
s = 1
omega_c = 0.5

nof_coefficients = 50

# Ortpol internal discretization parameter
orthpol_ncap = 10000


# Domain of the spectral density (choosing inf here will use orthpols internal cutoff)
domain = [0.0, np.inf]

J = lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-(x/omega_c))

c0, omega, t, info = \
    mapping.chain.get(J, domain, nof_coefficients, disc_type='bsdo', orthpol_ncap=orthpol_ncap)

print('System to bath coupling: ', c0)


plt.figure(1)
plt.plot(omega, '.')
plt.title('Bath energies $\\omega$')
plt.xlabel('Site index')
plt.figure(2)
plt.plot(t, '.')
plt.title('Bath couplings $t$')
plt.xlabel('Bond index')
plt.show()