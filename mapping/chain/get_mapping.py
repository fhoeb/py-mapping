from mapping.chain.mapping.householder import HouseholderMapping
from mapping.chain.mapping.lanczos import LanczosMapping
from mapping.chain.mapping.lanczos_bath import LanczosBathMapping
from mapping.chain.mapping.scipy_hessenberg import ScipyHessenbergMapping


def get_mapping(discretized_bath, mapping_type='lan_bath', max_nof_coefficients=100, low_memory=True, stable=False):
    """
        Factory function for Mapping objects
    :param discretized_bath: Discretized bath object
    :param mapping_type: Type of the tridiagonalization to be used. Allowed are:
                        'hou' for Householder transformation of bath and dummy system (full storage)
                        'lan' for Lanczos tridiagonalization of bath and dummy system (sparse)
                        'lan_bath' for Lanczos tridiagonalization of the bath only (sparse)
                        'sp_hes' for using the scipy hessenberg method for bath and dummy system (full storage)
    :param max_nof_coefficients: Maximum number of coefficients
    :param low_memory: Uses a more memory efficient version of the lanczos algorithms, which may be slightly slower
    :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                   some stability problems encountered with Lanczos tridiagonalization
    :return: A Mapping object
    """
    if mapping_type == 'hou':
        return HouseholderMapping(discretized_bath, max_nof_coefficients=max_nof_coefficients)
    elif mapping_type == 'lan':
        return LanczosMapping(discretized_bath, max_nof_coefficients=max_nof_coefficients, low_memory=low_memory,
                              stable=stable)
    elif mapping_type == 'lan_bath':
        return LanczosBathMapping(discretized_bath, max_nof_coefficients=max_nof_coefficients, low_memory=low_memory,
                                  stable=stable)
    elif mapping_type == 'sp_hes':
        return ScipyHessenbergMapping(discretized_bath, max_nof_coefficients=max_nof_coefficients)
    else:
        print('Unrecognized mapping type')
        raise AssertionError
