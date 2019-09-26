import numpy as np


def get_null_space(A):
    """
        Computes the Null space (all the span of vectors v for which dot(A, v) = 0 holds)
    :param A: Matrix from which to compute the null space
    :return: A matrix q, whose columns make up the null space of A
    """
    m, n = A.shape
    # TODO: Convert assertion into proper exception or error message here
    assert n > m
    q, r = np.linalg.qr(A, mode='complete')
    null_space = q[:, n-m:]
    return null_space


def get_orthogonal_vector(A, mode='qr'):
    """
    :param A: Vectors arranged as columns of a matrix A, relative to which an orthogonal vector is found.
    :param mode: 'qr' uses QR decomposition for the calculation (no other method is currently supported)
    :return: A vector, which is orthogonal to the vectors, which make up the columns of A
    """
    if mode == 'qr':
        null_space = get_null_space(A)
        return null_space[:, -1]
    else:
        print('Unsupported mode')
        raise AssertionError