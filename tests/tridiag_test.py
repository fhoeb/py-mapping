import mapping
import pytest
import numpy as np
from scipy.linalg import hessenberg


# dim must be low due to the intrinsic instability of the lanczos algorithm
@pytest.mark.fast
@pytest.mark.parametrize('dim', [2, 5, 10])
@pytest.mark.parametrize('seed', [5, 102])
@pytest.mark.parametrize('method', ['lan', 'sp_hes', 'hou'])
def test_tridiag(dim, seed, method):
    """
        Test for the tridiagonalization algorithms
    """
    acc = 1e-10
    np.random.seed(seed)
    A = np.random.rand(dim, dim)
    A_sym = 1 / 2 * (A + A.T)
    A_tri, trafo_ref = hessenberg(A_sym, calc_q=True)
    diag_ref, offdiag_ref = np.diag(A_tri), np.diag(A_tri, k=-1)
    diag, offdiag, info = mapping.tridiag(A_sym, view=None, method=method, get_trafo=True, residual=True)
    assert np.max(np.abs(diag_ref - diag) / np.abs(diag_ref)) < acc
    assert np.max(np.abs(np.abs(offdiag_ref) - offdiag) / np.abs(offdiag_ref)) < acc
    trafo = info['trafo']
    A_trafo = trafo.T @ A_sym @ trafo
    diag_trafo, offdiag_trafo = np.diag(A_trafo), np.diag(A_trafo, k=-1)
    assert np.max(np.abs(diag - diag_trafo) / np.abs(diag)) < acc
    assert np.max(np.abs(offdiag - offdiag_trafo) / np.abs(offdiag)) < acc