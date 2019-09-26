"""
    A clas and interface function for permuting star coefficients
"""
import numpy as np


class Sorting:
    """
        Class, which can be used to create sorting objects for star coefficients
    """
    def __init__(self):
        self.sort = None

    def select(self, sort_by):
        """
            Select the kind of sorting
        :param sort_by: See sort_star_coefficients for a detailed explanation
        :return:
        """
        if sort_by is None:
            self.sort = None
        elif sort_by == 'inv':
            self.sort = self.invert
        elif sort_by == 'xi_a':
            self.sort = self.xi_ascending
        elif sort_by == 'gamma_a':
            self.sort = self.gamma_ascending
        elif sort_by == 'xi_d':
            self.sort = self.xi_descending
        elif sort_by == 'gamma_d':
            self.sort = self.gamma_descending
        elif sort_by == 'abs_xi_a':
            self.sort = self.abs_xi_ascending
        elif sort_by == 'abs_gamma_a':
            self.sort = self.abs_gamma_ascending
        elif sort_by == 'abs_xi_d':
            self.sort = self.abs_xi_descending
        elif sort_by == 'abs_gamma_d':
            self.sort = self.abs_gamma_descending
        else:
            print('Unrecognized sorting type')
            raise AssertionError

    @staticmethod
    def invert(gamma, xi):
        return np.arange(len(gamma)-1, -1, -1, dtype=int)

    @staticmethod
    def abs_xi_ascending(gamma, xi):
        return np.argsort(np.abs(xi), kind='mergesort')

    @staticmethod
    def abs_gamma_ascending(gamma, xi):
        return np.argsort(np.abs(gamma), kind='mergesort')

    @staticmethod
    def abs_xi_descending(gamma, xi):
        return np.argsort(np.abs(xi), kind='mergesort')[::-1]

    @staticmethod
    def abs_gamma_descending(gamma, xi):
        return np.argsort(np.abs(gamma), kind='mergesort')[::-1]

    @staticmethod
    def xi_ascending(gamma, xi):
        return np.argsort(xi, kind='mergesort')

    @staticmethod
    def gamma_ascending(gamma, xi):
        return np.argsort(gamma, kind='mergesort')

    @staticmethod
    def gamma_descending(gamma, xi):
        return np.argsort(gamma, kind='mergesort')[::-1]

    @staticmethod
    def xi_descending(gamma, xi):
        return np.argsort(xi, kind='mergesort')[::-1]


def sort_star_coefficients(gamma, xi, sort_by=None):
    """
        Sorts the star couplings and bath energies, in a way specified by sort_by
    :param gamma: Array of star couplings
    :param xi: Array of star bath energies
    :param sort_by: String (or None), which specifies the kind of sorting. Supported are:
                    'inv': Inverts both arrays
                    'xi_a': Permutes both arrays, such that the energies are in ascending order
                    'gamma_a': Permute both arays, such that the couplings are in ascending order
                    'xi_d': Permutes both arrays, such that the energies are in descending order
                    'gamma_d': Permutes both arrays, such that the couplings are in descending order
                    'abs_xi_a': Permutes both arrays, such that the absolute values of the energies are in ascending
                                order
                    'abs_gamma_a': Permute both arays, such that the absolute values of the couplings are in
                                   ascending order
                    'abs_xi_d': Permutes both arrays, such that the absolute values of the energies are in descending
                                order
                    'abs_gamma_d': Permutes both arrays, such that the absolute values of the couplings are in
                                   descending order
                    Function is a NOP if sort_by is None

    :return: Sorted gamma (star couplings), sorted xi (star bath energies)
    """
    if sort_by is None:
        return gamma, xi
    else:
        sorting = Sorting()
        sorting.select(sort_by)
        sorted_indices = sorting.sort(gamma, xi)
        return gamma[sorted_indices], xi[sorted_indices]
