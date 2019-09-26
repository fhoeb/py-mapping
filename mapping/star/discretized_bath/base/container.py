"""
    Container class, which can be used to store pre-calculated couplings/energies, such that they can be used
    for a chain mapping via mapping objects
"""
from mapping.star.discretized_bath.base.base import BaseDiscretizedBath
from mapping.star.discretized_bath.base.eocoeff import EOFCoefficients


class DiscretizedBath(BaseDiscretizedBath):
    def __init__(self, gamma, xi):
        """
            Container for pre-calculated coefficients for a discretized bath
        """
        assert len(gamma) == len(xi)
        super().__init__('asym', len(gamma),  None)
        self.gamma_buf = gamma
        self.xi_buf = xi
        self._next_n = len(gamma)

    def compute_next(self, nof_coefficients):
        """
            Calculates the next *nof_coefficients* coefficients and places them in the Buffer
        :param nof_coefficients: Number of coefficients to calculate (starting from n=next_n)

        """
        raise EOFCoefficients(self.next_n)
