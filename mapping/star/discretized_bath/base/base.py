"""
    Base class for discretized bath objects
"""
import numpy as np
from math import fsum
from mapping.star.discretized_bath.base.eocoeff import EOFCoefficients
from mapping.star.discretized_bath.stopcoeff import StopCoefficients
from mapping.utils.sorting import sort_star_coefficients


class BaseDiscretizedBath:
    def __init__(self, discretization_type, max_nof_coefficients, discretization_callback):
        """

        :param discretization_type: Either symmetric ('sym') or asymmetric ('asym')
        :param max_nof_coefficients: Argument for size of the coefficient buffers. The actual size of the
                                     buffers depends also on whether the bath is symmetric or not.
                                     If asymmetric, max_nof_coefficients equals the buffer size.
                                     If symmetric, max_nof_coefficients is only half the buffer size
        :param discretization_callback: Callback, which fills the buffers with coefficients and updates the internal
                                        counter _next_n, which specifies to what amount the buffers are currently
                                        filled. Again, taking into account the size of symmetric discretization buffers
        """
        assert (discretization_type == 'asym') or (discretization_type == 'sym')
        self.is_symmetric = False if discretization_type == 'asym' else True
        self.type = discretization_type
        self.bufsize = 2 * max_nof_coefficients if self.is_symmetric else max_nof_coefficients
        self.gamma_buf = np.empty(self.bufsize)
        self.xi_buf = np.empty(self.bufsize)
        self.max_nof_coefficients = max_nof_coefficients
        self.discretization_callback = discretization_callback
        # Index of the next xi_n/gamma_n coefficients which are generated
        self._next_n = 0

    @property
    def next_n(self):
        """
            Index of the next coefficient(s, for symmetric) to be calculated
        """
        return self._next_n

    def _update_next_n(self, step):
        """
            Updates next_n by step
        """
        self._next_n += step

    def _set_next_n(self, next_n):
        """
            Sets next_n to the value specifiec by the argument
        """
        self._next_n = next_n

    @property
    def gamma(self):
        """
            Returns a view of the gamma (couplings) buffer with the coefficients calculated up to now.
            Contains twice as many coefficients for the symmetric as for the asymmetric discretization
        """
        if not self.is_symmetric:
            return self.gamma_buf[:self._next_n]
        else:
            return self.gamma_buf[:2*self._next_n]

    @property
    def xi(self):
        """
            Returns a view of the xi (energies) buffer with the coefficients calculated up to now.
            Contains twice as many coefficients for the symmetric as for the asymmetric discretization
        """
        if not self.is_symmetric:
            return self.xi_buf[:self._next_n]
        else:
            return self.xi_buf[:2*self._next_n]

    @property
    def eta_0(self):
        """
            Returns sum_i gamma_i (of the currently calculated coefficients)
        """
        return fsum(np.square(self.gamma))

    def show(self):
        """
        :return: View of the couplings (gamma), View of the energies (xi)
        """
        return self.gamma, self.xi

    def get(self):
        """
        :return: Copy of the couplings (gamma), Copy of the energies (xi)
        """
        return self.gamma.copy(), self.xi.copy()

    def get_sorted(self, sort_by=None):
        """
        :return: Copy of the couplings (gamma), Copy of the energies (xi). Both sorted as specified by the
                 sort_by argument (see utils.sorting.sort_star_coefficients)
        """
        return sort_star_coefficients(self.gamma, self.xi, sort_by=sort_by)

    def sort(self, sort_by=None):
        """
            Sorts the filled parts of the coupling and energy buffers in place. Both are sorted as specified by the
            sort_by argument (see utils.sorting.sort_star_coefficients)
        """
        if not self.is_symmetric:
            self.gamma_buf[:self._next_n], self.xi_buf[:self._next_n] = \
                sort_star_coefficients(self.gamma_buf[:self._next_n], self.xi_buf[:self._next_n], sort_by=sort_by)
        else:
            self.gamma_buf[:2*self._next_n], self.xi_buf[:2*self._next_n] = \
                sort_star_coefficients(self.gamma_buf[:2*self._next_n], self.xi_buf[:2*self._next_n], sort_by=sort_by)

    def reset(self):
        """
            Resets the object, to set from which index onwards the next coefficients are calculated.
        """
        self._next_n = 0

    def compute_next(self, nof_coefficients):
        """
            Calls the discretization_callback function with nof_coefficients.
            Does error handling for the discretization_callback passed by the child class to the base constructor
        :param nof_coefficients: Number of coefficients to calculate (starting from n=next_n)

        """
        try:
            self.discretization_callback(self._next_n + nof_coefficients)
        except ZeroDivisionError:
            print('Error: Div/0. Cannot calculate more coefficients!')
            raise EOFCoefficients(self._next_n)
        except FloatingPointError:
            print('Floating Point Error. Cannot calculate more coefficients!')
            raise EOFCoefficients(self._next_n)
        except StopCoefficients:
            raise EOFCoefficients(self._next_n)
        except IndexError:
            print('Error: Reached buffer limit. Cannot calculate more coefficients!')
            raise EOFCoefficients(self._next_n)

    def compute_all(self):
        """
            Fills the buffers with coefficients until max_nof_coefficients (or 2*max_nof_coefficients for symmetric
            baths)
        """
        if self._next_n < self.max_nof_coefficients:
            self.compute_next(nof_coefficients=self.max_nof_coefficients - self._next_n)

    def compute_until(self, nof_coefficients):
        """
            Computes coefficients until nof_coefficients coefficients are in the buffers (or 2*nof_coefficients
            for symmetric discretizations)
        """
        if self._next_n < nof_coefficients:
            self.compute_next(nof_coefficients=nof_coefficients - self._next_n)
