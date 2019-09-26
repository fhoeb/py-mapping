"""
    Base class for asymmatric bath discretizations
"""
from mapping.star.discretized_bath.base.base import BaseDiscretizedBath


class BaseDiscretizedAsymmetricBath(BaseDiscretizedBath):
    """
        Generation of coefficients for logarithmic discretization of bosonic baths
    """
    def __init__(self, discretization_callback, max_nof_coefficients=100):
        """
        :param discretization_callback: Callback, which fills the buffers with coefficients and updates the internal
                                        counter _next_n, which specifies to what amount the buffers are currently
                                        filled.
                                        signature:
                                        f(stop_n)
                                        where stop_n specifies what next_n must be set to after successful computation
                                        The number of coefficients to calculate are thus stop_n - next_n
                                        for the asymmetric bath.
        :param max_nof_coefficients: Size of the pre-allocated buffers
        """
        super().__init__('asym', max_nof_coefficients, discretization_callback)
