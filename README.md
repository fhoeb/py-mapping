## A quantum bath discretization library for python

* Supports arbitrary bath spectral densities J(x) 
* Support for chain (next nearest neighbor coupling) and star (long range coupling) geometry coefficient
generation
* Support for various kinds of integration (Custom/scipy Quadrature rule implementations, Midpoint rule,
Trapezoidal rule) and tridiagonalization algorithms (Householder transformations, Lanczos algorithm)
* Supports the computation of Bath Spectral Density Orthogonal discretizations using orthpol (see for example: Chin et al., J. Math. Phys. 51, 092109 (2010) and de Vega et al.,  Phys. Rev. B 92, 155126 (2015)
* Support for logarithmic and simple custom discretizations, as well as convergence based methods for chain mappings

To install the latest stable version run

    pip install py-mapping


Required packages:

* numpy, scipy, py-orthpol

Optional (recommended) packages:

* mpmath

Supported Python versions:

* 3.5, 3.6, 3.7


#### Remark:
The convention we use for the definition of the spectral density does not include the factor of pi, 
which is used for example in Chin et al., J. Math. Phys. 51, 092109 (2010), but not in
de Vega et al.,  Phys. Rev. B 92, 155126 (2015)
This can be remedied by including the factor in the function definition of the spectral density

## Contributors

* Fabian Hoeb, <fabian.hoeb@uni-ulm.de>, [University of Ulm]


## License

Distributed under the terms of the BSD 3-Clause License (see [LICENSE](LICENSE)).
