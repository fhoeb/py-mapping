"""
    Factory function for discretized baths
"""
from mapping.star.discretized_bath.asymmetric_spquad import SpQuadDiscretizedAsymmetricBath
from mapping.star.discretized_bath.symmetric_spquad import SpQuadDiscretizedSymmetricBath
from mapping.star.discretized_bath.asymmetric_bsdo import BSDODiscretizedAsymmetricBath
from mapping.star.discretized_bath.asymmetric_midpt import MidptDiscretizedAsymmetricBath
from mapping.star.discretized_bath.symmetric_midpt import MidptDiscretizedSymmetricBath
from mapping.star.discretized_bath.asymmetric_trapz import TrapzDiscretizedAsymmetricBath
from mapping.star.discretized_bath.symmetric_trapz import TrapzDiscretizedSymmetricBath
from mapping.star.discretized_bath.asymmetric_gk_quad import GKQuadDiscretizedAsymmetricBath
from mapping.star.discretized_bath.symmetric_gk_quad import GKQuadDiscretizedSymmetricBath
from mapping.star.discretized_bath.asymmetric_quad import QuadDiscretizedAsymmetricBath
from mapping.star.discretized_bath.symmetric_quad import QuadDiscretizedSymmetricBath
from mapping.star.discretized_bath.asymmetric_mean import MeanDiscretizedAsymmetricBath


def get_discretized_bath(J, domain, max_nof_coefficients=100, disc_type='sp_quad', interval_type='lin', **kwargs):
    """
        Factory function for bath discretizations. Returns either a DiscretizedBosonicBath or DiscretizedFermionicBath
        object depending on the domain. If the domain is purely positive, the bosonic discretization variant is chosen,
        if the domain is mixed positive and negative, the fermionic discretization variant is chosen.
    :param J: Spectral density. A function defined on 'domain', must be >0 in the inner part of domain.
    :param domain: List/tuple of two elements for the left and right boundary of the domain of J (e.g. [a, b].
                   May either be a, b >= 0 or a < 0 and b > 0
    :param max_nof_coefficients: Size of the buffers which hold gamma and xi coefficients (maximum number of
                                 these coefficients that can be calculated) for asymmetric discretization.
                                 For symmetric discretization, the buffers are twice as large as this value, since
                                 coefficients are always calculated in pairs, for the positive and negative
                                 part of the domain respectively.
    :param disc_type: Type of discretization procedure to be used. Currently supported are:
                     'spquad' for non vectorized quadrature rule direct discretization with scipy quad
                     'bsdo' for bsdo discretization with orthpol
                     'midpt' for midpoint rule direct discretization
                     'trapz' for trapezoidal rule direct discretization
                     'quad' for vectorized Gauss Legendre quadraure rule direct discretization with fixed order
                     'gk_quad' for adaptive vectorized Gauss-Kronrod quadraure rule direct discretization
                     'mean': for heuristic mean direct discretization (interval_type is irrelevant for this option, uses
                             non vectorized scipy quadrature), currently only available for asymmetric discretizations
                             (see de Vega et al.,  Phys. Rev. B 92, 155126 (2015) for details on the method)
                     Default is sp_quad
    :param interval_type: Type of interval to be used for the integrations (irrelevant for bsdo discretization)
                          'lin': Linearly space intervals
                          'log': Logarithmically spaced intervals
                          'custom_fcn': For using a custom function:
                                        float fcn(n)
                                        For bosonic and asymmetric fermionic spectral densities returns the
                                        corresponding energy. Only for asymmetric discretizations.
                                        n is the index of the grid point. Starting with n=0, ending with
                                        n=max_nof_coefficients-1. The intervals are then taken between the grid points.
                          'custom_arr': For using a custom grid for the discretization (a numpy array)
                                        For asymmetric discretizations only, must contain all grid-points between which
                                        the intervals for direct discretization are taken.
                                        Only for asymmetric discretization. max_nof_coefficients is then set to the size
                                        of this array
                          The points obtained by the custom_fcn and custom_arr types are sorted in ascending order,
                          before they are used for the discretization.
                          Default is 'lin'
    :param kwargs: Several different possible keyword arguments for selected discretization and interval types:
                  'symmetric': (bool) Uses symmetric (around 0) discretization procedure for intervals with
                               domain[0] < 0 < domain[1]. Default is False
                  'Lambda': (float > 1) Logarithmic discretization parameter (relevant only for 'log' interval_type),
                            default is 1.1
                  'interval_fcn': (callable) Function for the custom_fcn interval type, only for asymmetric
                                  discretization
                  'interval_arr': (numpy array) Array for the custom_arr interval type, only for asymmetric
                                  discretization
                  'orthpol_ncap': (int) Accuracy parameter for bsdo discretization for orthpol, default is 60000
                  'order': (int) Order of the fixed order vectorized quadrature rule, default is 30
                  'ignore_zeros': (bool) Choose to ignore (numerically) zero parts of the spectral density,
                                  Couplings in these parts will be 0, the corresponding direct discretization energies
                                  are then also set to 0,  default is False
                  'epsrel': (float) Relative error tolerance for scipy quad and gk_quad, default is 1e-11
                  'epsabs': (float) Absolute error tolerance for scipy quad and gk_quad, default is 1e-11
                  'limit': (int) Limit of the number of interval subdivisions for scipy quad, default is 1e-11
                  'mp_dps': mpmath dps (arbitrary precision decimal digits) for the star bsdo
                            diagonalization, default is 30
                  'force_sp': force the usage of scipy diagonalization for the star bsdo map, regardless of whether
                              an installed mpmath was detected or not, default is False
    :return: A constructed discretized bath object.
    """
    try:
        symmetric = kwargs['symmetric']
    except KeyError:
        symmetric = False
    if interval_type == 'custom_arr':
        try:
            arr = kwargs['interval_arr']
            max_nof_coefficients = len(arr)
        except KeyError:
            pass
    if 0 <= domain[0] <= domain[1] or not symmetric:
        if disc_type == 'sp_quad':
            return SpQuadDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                   interval_type=interval_type, **kwargs)
        elif disc_type == 'bsdo':
            return BSDODiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients, **kwargs)
        elif disc_type == 'midpt':
            return MidptDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                  interval_type=interval_type,
                                                  **kwargs)
        elif disc_type == 'trapz':
            return TrapzDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                  interval_type=interval_type,
                                                  **kwargs)
        elif disc_type == 'quad':
            return QuadDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                 interval_type=interval_type,
                                                 **kwargs)
        elif disc_type == 'gk_quad':
            return GKQuadDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                   interval_type=interval_type, **kwargs)
        elif disc_type == 'mean':
            return MeanDiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients, **kwargs)
        else:
            print('Unsupported discretization type')
            raise AssertionError
    elif domain[0] < 0 < domain[1]:
        if disc_type == 'sp_quad':
            return SpQuadDiscretizedSymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                  interval_type=interval_type,
                                                  **kwargs)
        elif disc_type == 'bsdo':
            return BSDODiscretizedAsymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients, **kwargs)
        elif disc_type == 'midpt':
            return MidptDiscretizedSymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                 interval_type=interval_type,
                                                 **kwargs)
        elif disc_type == 'trapz':
            return TrapzDiscretizedSymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                 interval_type=interval_type,
                                                 **kwargs)
        elif disc_type == 'quad':
            return QuadDiscretizedSymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                interval_type=interval_type,
                                                **kwargs)
        elif disc_type == 'gk_quad':
            return GKQuadDiscretizedSymmetricBath(J, domain, max_nof_coefficients=max_nof_coefficients,
                                                  interval_type=interval_type,
                                                  **kwargs)
        else:
            print('Unsupported discretization type')
            raise AssertionError
    else:
        print('Unsupported combination of domain and symmetry')
        raise AssertionError
