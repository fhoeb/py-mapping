# Brief documentation of the project structure and naming conventions

## Project structure:

The project can be subdivided into three sub-modules:
* The implementation of the tridiagonalization algorithms, which is itself loosely object based
* The implementation of star geometry discretization algorithms. Based on the BaseDiscretizedBath class and its inheritors
* The star to chain mapping either directly via orthpol or via the BaseMapping class and its inheritors, which use discretized bath objects

#### Tridiagonalization 
The tridiagonalization classes are independent, but the interface is mostly compatible with some exceptions,
particularly for the Lanczos algorithm implementations.
All of them provide the ability to tridiagnalize sub-matrices (via views) of a full matrix buffer, which is used
for the convergence type of chain mapping.

Furthermore they all use the same way to compute the residual of the tridiagonalization (in info dicts under the
keyword 'res'). Residuals are returned as tuples with (Frombenius norm of (V.T @ V - id), max(|V.T @ V - id|)
where V is the transformation matrix of the tridiagonalization for the matrix A (T = V.T A V).

#### Star sub-module
The basis of the implementation is the container class BaseDiscretizedBath, which hides the buffers that
contain calculated coefficients and provide interfaces to inspect and fill those buffers.

Direct subclasses are the SymmetricDiscretizedBath and AsymmetricDiscretizedBath classes, where the
symmetric one uses twice the size of the buffers and is supposed to be filled with two coefficients per step to
ensure a symmetric discretization. Both subclasses use the same counters to check how far the buffers are filled
and what the next index to be filled would be. 

One must take care when handling the symmetric baths. For example:
The max_nof_coefficients argument does indeed specify the buffer size for asymmetric baths. In the symmetric case
it specifies only half the buffer size and each computation step leads to two new generated coefficients.

Each subclass of the SymmetricDiscretizedBath and AsymmetricDiscretizedBath classes specifies an implementation
of a star type discretization. For example asymmetric_bsdo implements BSDO type discretization, 
symmetric_gk_quad implements symmetric direct discretization using a vectorized Gauss-Kronrod Quadrature for the
computation of the integrals.

#### Chain sub-module
One can further subdivide the chain sub-module in two distinct parts. One is an implementation of the TEDOPA/BSDO
algorithm (as found in Chin et al., J. Math. Phys. 51, 092109 (2010), for example). The other implements a
class called BaseMapping and inheritors, which are built to utilize the various different tridiagonalization algorithms.

The TEDOPA part straightforwardly uses py-orthpol for the computation of the monic polynomial recurrence coefficients.

The BaseMapping related classes/interfaces then map arbitrary star geometry coefficients to chain geometry coefficients,
either for a fixed number of star coefficients or by using a convergence condition, on the chain coefficients.


#### Conversions
There are three direct conversion functions, which are implemented in utils.convert. 

The convert_star_to_chain function
uses diagonalization (either float precision via scipy or arbitrary precision via mpmath if an installation is detected)
to convert chain coefficients to star coefficients. 

The convert_star_to_chain function uses scipy's hessenberg 
function to calculate the equivalent chain coefficients from the star coefficients (float precision, but rather stable).

Finally convert_star_to_chain_lan uses Lanczos tridiagonalization (which is faster und surprisingly stable in many
cases). 

The star to chain conversion methods return a residual in the info dict under 'res', which is of the form described in
the Tridiagonalization or Notation section, that allows one to check the validity of the results.

## Notation and conventions throughout the project
For the star coefficients:
* gamma denotes the couplings from the system to the bath sites
* xi denotes the bath site energies

For the chain coefficients:
* c0 denotes the coupling from the system site to the start of the chain
* omega denotes the  bath site energies
* t denotes the bath-bath couplings 

Wherever possible, info dictionaries are returned, which may contain an entry called 'res', which specifies the
residual of any tridiagonalizations that have been performed. 
The residual has the form: tuple(Frombenius norm of (V.T @ V - id), max(|V.T @ V - id|),
where V is the transformation matrix of the tridiagonalization for the matrix A (T = V.T A V)


