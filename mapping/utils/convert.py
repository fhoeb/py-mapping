"""
    Module for conversions between star and chain geometry coefficients
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from math import fsum
try:
    from mpmath import mp, eigsy
except ImportError:
    print('WARNING: No installation of mpmath detected, this may result in inaccuracies in chain to star '
          'conversions')
    mp = None
    eigsy = None
from mapping.tridiag.scipy_hessenberg.full import ScipyHessenberg
from mapping.tridiag.lanczos.diag_low_memory import LowMemLanczosDiag
from mapping.utils.sorting import Sorting, sort_star_coefficients


def convert_chain_to_star(c0, omega, t, get_trafo=False, force_sp=False, mp_dps=30, sort_by=None):
    """
        Converts chain coefficients in the form c0, omega, t (system to bath coupling, bath energies,
        bath-bath couplings) into the equivalent star geometry coefficients gamma, xi (star system to bath coupling,
        star bath energies) by using diagonalization with either arbitrary precision mpmath if the library is installed
        or scipy eigh_tridiagonal in float precision
    :param c0: System to bath coupling float
    :param omega: Bath energies (numpy array)
    :param t: Bath-bath couplings (numpy array)
    :param get_trafo: If the transformation between the chain and the star should be returned or not
                      This matrix is only for the omega/t coefficients
    :param force_sp: Force the use of the scipy method eigh_tridiagonal, even if mpmath is installed
    :param mp_dps: Decimals, which mpmath uses for the computation
    :return: gamma (star system to bath coupling), xi (star bath energies),
             info dict with the keys: 'trafo': Contains the transformation Matrix between the geometries
    """
    assert len(omega)-1 == len(t)
    info = dict()
    info['trafo'] = None
    if mp is None or force_sp:
        w, v = eigh_tridiagonal(omega, t)
        gamma = c0 * np.abs(v[0, :])
        xi = w
        if get_trafo:
            info['trafo'] = v
    else:
        mp.set_dps = mp_dps
        nof_coefficients = len(omega)
        A = np.zeros((nof_coefficients, nof_coefficients))
        drows, dcols = np.diag_indices_from(A)
        A[drows[:nof_coefficients], dcols[:nof_coefficients]] = omega
        rng = np.arange(nof_coefficients - 1)
        A[rng + 1, rng] = t
        A[rng, rng + 1] = t
        E, Q = mp.eigsy(mp.matrix(A.tolist()))
        xi = np.empty(nof_coefficients)
        gamma = np.empty(nof_coefficients)
        for i in range(A.shape[1]):
            xi[i] = float(E[i])
            gamma[i] = c0 * np.abs(float(Q[0, i]))
        if get_trafo:
            Q = np.array(Q.tolist(), dtype=np.float64)
            info['trafo'] = Q
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info


def convert_star_to_chain(gamma, xi, residual=True, get_trafo=False, positive=True, permute=None):
    """
        Converts star coefficients in the form gamma, xi (star system to bath coupling, star bath energies)
        into the equivalent chain geometry coefficients c0, omega, t (system to bath coupling, bath energies,
        bath-bath couplings) by using tridiagonalization with scipy's hessenberg method.
    :param gamma: Star system to bath coupling as numpy array
    :param xi: Star bath energies as numpy array
    :param residual: If set True, the residual for the tridiagoalization is computed and included in the info dict
    :param get_trafo: If the transformation between the star and chain should be returned or not.
                      The matrix is for the full coefficient matrix (including c0)
    :param positive: If set False the transformation matrix between star and chain is the one directly from
                     scipy, where the tridiagonal form may in general contain negative offdiagonals.
                     These are unphysical and the returned t-coefficients are absolute values of those.
                     If set True, the transformation amtrix is adapted to match the positive offdiagonals
    :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                    sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                    possible parameters). This may help increase numerical stability for the tridiagonalization.
    :returns: c0 (system to bath coupling), omega (bath energies), t (bath-bath couplings),
              info dict with the keys: 'trafo': Contains the transformation Matrix between the geometries
                                       'res': Contains the computed residual
    """
    assert len(gamma) == len(xi)
    if permute is not None:
        sorting = Sorting()
        sorting.select(permute)
        sorted_indices = sorting.sort(gamma, xi)
        xi = xi[sorted_indices]
        gamma = gamma[sorted_indices]
    ncap = len(gamma)
    A = np.zeros((ncap+1, ncap+1))
    drows, dcols = np.diag_indices_from(A)
    A[drows[1:ncap + 1], dcols[1:ncap + 1]] = xi
    A[0, 1:ncap + 1] = gamma
    A[1:ncap + 1, 0] = A[0, 1:ncap + 1]
    diag, offdiag, info = ScipyHessenberg(A).get_tridiagonal(residual=residual, get_trafo=get_trafo, positive=positive)
    return offdiag[0], diag[1::], offdiag[1:], info


def convert_star_to_chain_lan(gamma, xi, residual=True, get_trafo=False, stable=True, permute=None):
    """
        Converts star coefficients in the form gamma, xi (star system to bath coupling, star bath energies)
        into the equivalent chain geometry coefficients c0, omega, t (system to bath coupling, bath energies,
        bath-bath couplings) by using tridiagonalization with the Lanczos method for the bath only.
    :param gamma: Star system to bath coupling as numpy array
    :param xi: Star bath energies as numpy array
    :param residual: If set True, the residual for the tridiagoalization is computed and included in the info dict
    :param get_trafo: If the transformation between the star and chain should be returned or not.
                      This matrix is only for the omega/t coefficients
    :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                   some stability problems encountered with Lanczos tridiagonalization
    :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                    sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                    possible parameters). This may help increase numerical stability for the tridiagonalization.
    :returns: c0 (system to bath coupling), omega (bath energies), t (bath-bath couplings),
              info dict with the keys: 'trafo': Contains the transformation Matrix between the geometries
                                       'res': Contains the computed residual
    """
    assert len(gamma) == len(xi)
    if permute is not None:
        sorting = Sorting()
        sorting.select(permute)
        sorted_indices = sorting.sort(gamma, xi)
        xi = xi[sorted_indices]
        gamma = gamma[sorted_indices]
    c0 = np.sqrt(fsum(np.square(gamma)))
    diag, offdiag, info = LowMemLanczosDiag(xi, gamma / np.linalg.norm(gamma),
                                            stable=stable).get_tridiagonal(residual=residual, get_trafo=get_trafo)
    return c0, diag, offdiag, info