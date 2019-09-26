"""
    Base class for symmatric (around 0) bath discretizations
"""
from mapping.star.discretized_bath.base.base import BaseDiscretizedBath


class BaseDiscretizedSymmetricBath(BaseDiscretizedBath):
    """
        Generation of coefficients for the discretization of quantum baths, symmetric around 0
    """
    def __init__(self, discretization_callback, max_nof_coefficients=100):
        """
        :param discretization_callback: Callback, which fills the buffers with coefficients and updates the internal
                                        counter _next_n, which specifies to what amount the buffers are currently
                                        filled. As with max_nof_coefficients, the buffer is then actually filled up to
                                        2*_next_n, since coefficients are generated in pairs for symmetric
                                        discretization.
                                        signature:
                                        f(stop_n)
                                        where stop_n specifies what next_n must be set to after successful computation
                                        The number of coefficients to calculate are thus 2*stop_n - next_n
                                        for the symmetric bath. Where half of those are for the positive and
                                        the other half for the negative domain.
        :param max_nof_coefficients: Size argument of the pre-allocated buffers
                                     (actually only half the size, since for symmetric discretization coefficients
                                      are always generated in pairs)
        """
        super().__init__('sym', max_nof_coefficients, discretization_callback)
