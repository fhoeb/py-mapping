import numpy as np


def orth_residual(V):
    """
        Tests orthogonality of V by computing:
        the frombenius norm of (V^T @ V - id) and the maximum of |V^T @ V - id|
    :param V: n x m numpy ndarray with orthonormal columns
    :return: Resudal tuple as specified above
    """
    diff = V.T @ V - np.identity(V.shape[1])
    return np.linalg.norm(diff), np.max(np.abs(diff))