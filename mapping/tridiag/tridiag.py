"""
    Functions for the tridiagonalization of symmetric real matrices
"""
from mapping.tridiag.get_tridiag_solver import get_tridiag, get_tridiag_from_diag, get_tridiag_from_special_sparse


def tridiag(A, view=None, method='sp_hes', low_memory=True, cutoff=None, v0=None, stable=False, residual=True,
            get_trafo=False, positive=True):
    """
        Tridiagonalizes a real symmetric full-storage matrix A
    :param A: Matrix as numpy ndarray
    :param view: My take a view of the matrix first. If passed should be an integer, that indicates the dimension of the
                 quadratic sub-matrix to tridiagonalize staring from the top left element of the matrix,
    :param method: Method to use for the tridiagonalization. Allowed are
                   'sp_hes': for the scipy internal hessenberg method
                   'hou': for Householder transformations
                   'lan': for the Lanczos algorithm
    :param low_memory: For Lanczos only: Uses a more memory saving variant of the algorithm
    :param cutoff: How much of the matrix (or possibly the sub-matrix) A should be tridiagonalized.
                   Integer that indicates the desired number of diagonal elements of the resulting tridiagonal matrix
    :param v0: For Lanczos only: Starting vector. If None, it uses (1 0 ... 0).T. Must be normalized to 1
    :param stable: For Lanczos only: If a stable summation algorithm should be used (WARNING: slow)
    :param residual: Calculates the residual of the tridiagonalization. This uses extra memory and overrides
                     low_memory if selected. See utils.residual.orth_residual for details
    :param get_trafo: Returns the corresponding transformation matrix
    :param positive: For Householder transformations and the scipy internal hessenberg method, there may be negative
                     off-diagonal elements. This ensures, that a transformation is chosen, which guarantees positive
                     off-diagonal elements
    :return: diagonal elements (1d numpy array), off-diagonal elements (1d numpy array), info dict with entries
             'res' for the residual, 'trafo' for the transformation matrix U (T = U^T A U)
    """
    generator = get_tridiag(A, view=view, method=method, low_memory=low_memory, max_cutoff=cutoff,
                            v0=v0, stable=stable)
    if method == 'sp_hes' or method == 'hou':
        return generator.get_tridiagonal(cutoff=cutoff, residual=residual, get_trafo=get_trafo, positive=positive)
    else:
        return generator.get_tridiagonal(cutoff=cutoff, residual=residual, get_trafo=get_trafo)


def tridiag_from_diag(diag, view=None, low_memory=True, cutoff=None, v0=None, stable=False, residual=True,
                      get_trafo=True):
    """
        Same as tridiag but optimized for diagonal matrices. diag should contain the diagonal elements
        as numpy array. Always used Lanczos algorithm and thus does not require the positive keyword.
    """
    generator = get_tridiag_from_diag(diag, view=view, low_memory=low_memory, max_cutoff=cutoff,
                                      v0=v0, stable=stable)
    return generator.get_tridiagonal(cutoff=cutoff, residual=residual, get_trafo=get_trafo)


def tridiag_from_special_sparse(side, diag, view=None, low_memory=True, cutoff=None, v0=None, stable=False,
                                residual=True, get_trafo=True):
    """
        Same as tridiag but optimized for a special subclass of sparse matrices with the following properties:
        / 0 s1 s2 ..  \\
        |s1 d1 0  ..  |
        |s2 0 d2  ..  |
        \\: : :      /
        side contains the si (the elements of the top and left edge of the matrix) and
        diag contains the diagonal elements starting from element [1, 1] as 1d numpy arrays.
        Always uses Lanczos algorithm and thus does not require the positive keyword.
    """
    generator = get_tridiag_from_special_sparse(side, diag, view=view, low_memory=low_memory, max_cutoff=cutoff,
                                                v0=v0, stable=stable)
    return generator.get_tridiagonal(cutoff=cutoff, residual=residual, get_trafo=get_trafo)
