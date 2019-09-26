"""
    Factory function for tridiagonal solver objects
"""
from mapping.tridiag.householder.full import Householder
from mapping.tridiag.lanczos.full import Lanczos
from mapping.tridiag.scipy_hessenberg.full import ScipyHessenberg
from mapping.tridiag.lanczos.full_low_memory import LowMemLanczos
from mapping.tridiag.lanczos.diag import LanczosDiag
from mapping.tridiag.lanczos.diag_low_memory import LowMemLanczosDiag
from mapping.tridiag.lanczos.special_sparse import LanczosSpecialSparse
from mapping.tridiag.lanczos.special_sparse_low_memory import LowMemLanczosSpecialSparse


def get_tridiag(A, view=None, method='sp_hes', low_memory=True, max_cutoff=None, v0=None, stable=False):
    """
        Returns a tridiag solver object. The parameters which are also present on the tridiag function
        serve the identical purpose. Returns full-storage solvers only
        (sither one of the following: scipy hessenberg/ Householder/ Lanczos/ LowMemLanczos)
    """
    if method == 'sp_hes':
        return ScipyHessenberg(A, view=view)
    elif method == 'hou':
        return Householder(A, view=view)
    elif method == 'lan':
        if low_memory:
            return LowMemLanczos(A, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)
        else:
            return Lanczos(A, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)
    else:
        raise AssertionError('Unknown tridiagonalizaton method')


def get_tridiag_from_diag(diag, view=None, low_memory=True, max_cutoff=None, v0=None, stable=False):
    """
        Returns a tridiag solver object. The parameters which are also present on the tridiag_from_diag function
        serve the identical purpose. Returns sparse solvers only (either one of the following LowMemLanczosDiag/
        LanczosDiag)
    """
    if low_memory:
        return LowMemLanczosDiag(diag, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)
    else:
        return LanczosDiag(diag, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)


def get_tridiag_from_special_sparse(side, diag, view=None, low_memory=True, max_cutoff=None, v0=None, stable=False):
    """
        Returns a tridiag solver object. The parameters which are also present on the tridiag_from_special_sparse
        function serve the identical purpose. Returns sparse solvers only (either one of the following
        LowMemLanczosSpecialSparse/ LanczosSpecialSparse)
    """
    if low_memory:
        return LowMemLanczosSpecialSparse(side, diag, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)
    else:
        return LanczosSpecialSparse(side, diag, view=view, max_cutoff=max_cutoff, v0=v0, stable=stable)
