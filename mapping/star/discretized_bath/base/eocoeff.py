"""
    Exception, which is thrown if no more coefficients can be calculated (either due to a full buffer or due to
    div/0 exceptions)
"""


class EOFCoefficients(Exception):
    def __init__(self, nof_calc_coeff):
        """
            Constructor
        :param nof_calc_coeff: Number of successfully calculated discretization coefficients
        """
        self.nof_calc_coeff = nof_calc_coeff
