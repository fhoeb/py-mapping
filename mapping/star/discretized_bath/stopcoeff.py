"""
    Exception, which can be thrown, if the number of coefficients of the discretization is intrinsically finite
    and no more can be calculated
"""


class StopCoefficients(Exception):
    pass
