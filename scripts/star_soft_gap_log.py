import numpy as np
import matplotlib.pyplot as plt
import mapping


# Ohmic spectral density parameters:
Delta_0 = 0.5
r = 1

nof_coefficients = 50

# Logarithmic discretization parameter
Lambda = 1.1

# Chain coefficient convergence parameter:
max_ncap = 2000
min_ncap = 50

# Domain of the spectral density
domain = [-1.0, 1.0]

J = lambda x: Delta_0 * np.abs(x) ** r

c0, omega, t, info = \
    mapping.chain.get_from_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                                       interval_type='log', mapping_type='sp_hes',
                                       permute=None, residual=True, ignore_zeros=True,
                                       Lambda=Lambda, symmetric=True,
                                       max_ncap=max_ncap, min_ncap=min_ncap,
                                       stop_rel=1e-8, stop_abs=1e-8)

print(info)
print('System to bath coupling: ', c0)


plt.figure(1)
plt.plot(omega, '.')
plt.title('Bath energies $\\omega$')
plt.xlabel('Site index')
plt.ylim(-1, 1)
plt.figure(2)
plt.plot(t, '.')
plt.title('Bath couplings $t$')
plt.xlabel('Bond index')
plt.show()