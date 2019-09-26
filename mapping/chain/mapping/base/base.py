"""
    Base class for mapping a star to a chain. Either directly or via a convergence condition on the chain
    coefficients
"""

from mapping.star.discretized_bath.base.eocoeff import EOFCoefficients
from mapping.utils.convergence import check_convergence, check_array_convergence
from mapping.utils.sorting import Sorting
from mapping.utils.convert import convert_chain_to_star
from mapping.star.discretized_bath.base.container import DiscretizedBath
import numpy as np


class BaseMapping:
    def __init__(self, discretized_bath, tridiag, map_type, max_nof_coefficients):
        """
            Base Mapping Constructor
        :param discretized_bath: Discretized bath object (with or without pre-calculated coefficients)
        :param tridiag: Constructed tridiagonalizer (Lanczos/Hessenberg/Scipy-Internal or variants of those)
        :param map_type: Type of the mapping 'full' maps the bath and a dummy system. 'bath' only maps the bath
                         from star to chain geometry
        :param max_nof_coefficients: Maximal number of chain coefficients, that can be calculated by the mapper
                                    (buffer size)
        """
        assert discretized_bath.type == 'sym' or discretized_bath.type == 'asym'
        self.discretized_bath = discretized_bath
        self.is_asymmetric_bath = True if discretized_bath.type == 'asym' else False
        self.tridiag = tridiag
        self.type = map_type
        self.sorting = Sorting()
        self.max_nof_coefficients = max_nof_coefficients
        self.nof_calculated_coefficients = 0
        self._c0 = 0
        self.omega_buf = np.empty(max_nof_coefficients)
        self.t_buf = np.empty(max_nof_coefficients-1)

    @property
    def c0(self):
        """
            Coefficient for the coupling between the system and the start of the chain. If noe coefficients
            have been calculated so far it returns None
        """
        return self._c0 if self.nof_calculated_coefficients != 0 else None

    @property
    def omega(self):
        """
            Numpy array with the bath energies. If no coefficients have been calculated so far returns None
        """
        return self.omega_buf[:self.nof_calculated_coefficients] if self.nof_calculated_coefficients != 0 else None

    @property
    def t(self):
        """
            Numpy array with couplings between the bath sites. If no coefficients have been calculated so far returns None
        """
        return self.t_buf[:self.nof_calculated_coefficients-1] if self.nof_calculated_coefficients != 0 else None

    def show_coefficients(self):
        """
        :returns: c0, omega, t (views of the numpy arrays, None if nothing was calculated)
        """
        return self.c0, self.omega[:], self.t[:]

    def get_coefficients(self):
        """
        :return: c0, omega, t, (copies of the numpy arrays, None if nothing was calculated)
        """
        return self.c0, self.omega.copy() if self.omega is not None else None, \
               self.t.copy() if self.t is not None else None

    def get_discretized_bath_from_chain(self, get_trafo=False, force_sp=False, mp_dps=30):
        """
            Calculates the equivalent discretized bath of the currently calculated chain coefficients.
            Returns None if none have been calculated so far
            See utils.convert.convert_chain_to_star for an explanation of the parameters.
        :return: discretized bath object, info dict from conversion (see utils.convert.convert_chain_to_star)
        """
        if self.nof_calculated_coefficients == 0:
            return None, None
        else:
            gamma, xi = convert_chain_to_star(self.c0, self.omega, self.t, get_trafo=get_trafo, force_sp=force_sp,
                                              mp_dps=mp_dps)
            return DiscretizedBath(gamma, xi)

    def get_star_coefficients_from_chain(self, get_trafo=False, force_sp=False, mp_dps=30):
        """
            Calculates the equivalent star coefficients of the currently calculated chain coefficients.
            Returns None if none have been calculated so far
            See utils.convert.convert_chain_to_star for an explanation of the parameters
        :return: star couplings gamma, star energies xi, info dict from conversion
                 (see utils.convert.convert_chain_to_star)
        """
        if self.nof_calculated_coefficients == 0:
            return None, None, None
        else:
            return convert_chain_to_star(self.c0, self.omega, self.t, get_trafo=get_trafo, force_sp=force_sp,
                                         mp_dps=mp_dps)

    def _get_cutoff(self, nof_coefficients):
        """
            Helper method which returns the right matrix dimensions for the coefficient calculation depending
            on whether only the bath or the full dummy-system + bath are to be tridiagonalized
        :param nof_coefficients: Number of chain coefficients to be calculated
        :return: nof_coefficients if bath only, else nof_coefficients+1 (1 extra for the system, since
                 nof_coefficients only considers the bath sites)
        """
        if self.type == 'full':
            return nof_coefficients+1
        elif self.type == 'bath':
            return nof_coefficients
        else:
            print('Unrecognized tridiagonalisation type')
            raise AssertionError

    def _update_tridiag(self, ncap):
        """
            Update the tridiag object with the desired number of star coefficients ncap for the tridiagonalization
            (more specifically updates the matrix view inside the tridiag object)
            Is split into updates for symmetric discretizations and asymmetric ones (symmetric ones add twice the
            specified ncap, one in the positive one in the negative part of the domain)
        :param ncap: Number of coefficients with which the next tridiagonalization should be performed
        :return:
        """
        if self.is_asymmetric_bath:
            self._asymmetric_update(ncap)
        else:
            self._symmetric_update(ncap)

    def compute(self, nof_coefficients, ncap=None, permute=None, residual=False, get_trafo=False):
        """
            Central method, which computes the chain coefficients via tridigonalization from the discretized bath
            coefficients. Stores the result in the internal buffers of the object, accessible via
            c0, omega and t. Uses a fixed accuracy parameter/number of discretized coefficients ncap for the
            computation.
        :param nof_coefficients: Number of chain coefficients to be calculated
        :param ncap: Accuracy/Number of star coefficients to be used (must not be more than the discretized bath
                     passed upon construction can support)
        :param permute: If the star coefficients should be permuted before the tridiagonalization (essentially
                        sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                        possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                        specifically
        :param residual: If set True computes the residual for the tridiagonalization.
                         This may use extra memory!
        :param get_trafo: Returns the corresponding transformation matrix
        :return: info dict with entries: 'ncap': ncap used for the tridiagonalization.
                                         'res': Residual of the tridiagonalization
                                         'trafo': Transformation matrix if selected
        """
        if ncap is None:
            ncap = self.discretized_bath.max_nof_coefficients
        if ncap < nof_coefficients:
            print('Accuracy parameter set too low. Must be >= nof_coefficients. Automatically set ncap=nof_coefficients')
            ncap = nof_coefficients
        if nof_coefficients > self.discretized_bath.max_nof_coefficients:
            print('Number of coefficients to calculate must smaller or equal the size of the discretized bath!')
            raise AssertionError
        if self.discretized_bath.max_nof_coefficients < ncap:
            print('Error: Buffer of discretized bath not sufficient. Must fit at least ncap = ' + str(ncap) +
                  'elements')
            raise AssertionError
        if nof_coefficients > self.max_nof_coefficients:
            print('Error: Selected number of coefficients too high! Must increase max_nof_coefficients')
            raise AssertionError
        assert nof_coefficients <= ncap
        self.sorting.select(permute)
        cutoff = self._get_cutoff(nof_coefficients)
        try:
            self._update_tridiag(ncap)
        except EOFCoefficients as err:
            print('Limit for the calculation of gamma/xi coefficients reached. Using fixed max. accuracy of '
                  'ncap = ' + str(err.nof_calc_coeff))
            self._update_tridiag(err.nof_calc_coeff)
        alpha, beta, info = self.tridiag.get_tridiagonal(cutoff=cutoff, residual=residual,
                                                         get_trafo=get_trafo)
        info['ncap'] = ncap
        if self.type == 'full':
            self._c0, self.omega_buf[:nof_coefficients], self.t_buf[:nof_coefficients - 1] = \
                beta[0], alpha[1:], beta[1:]
        elif self.type == 'bath':
            self._c0, self.omega_buf[:nof_coefficients], self.t_buf[:nof_coefficients - 1] = \
                np.sqrt(self.discretized_bath.eta_0), alpha, beta
        else:
            print('Unrecognized tridiagonalisation type')
            raise AssertionError
        self.nof_calculated_coefficients = nof_coefficients
        return info

    def _asymmetric_update(self, ncap):
        """
            Updates tridiag for asymmetrically discretized baths. To be overridden by subclasses
        """
        pass

    def _symmetric_update(self, ncap):
        """
            Updates tridiag for symmetrically discretized baths. To be overridden by subclasses
        """
        pass

    def compute_from_convergence(self, nof_coefficients, min_ncap=10, max_ncap=2000, step_ncap=1,
                                 stop_rel=1e-10, stop_abs=1e-10, permute=None, residual=False, get_trafo=False):
        """
            Central method, which computes the chain coefficients via tridigonalization from the discretized bath
            coefficients. Stores the result in the internal buffers of the object, accessible via
            c0, omega and t. Determines the accuracy parameter/number of discretized coefficients ncap for the
            computation using a convergence condition for the chain coefficients omega, t. Iteratively uses
            more coefficients from the discretized bath in every step.
            Convergence condition considers all bath energies and couplings in every step
        :param nof_coefficients: Number of chain coefficients to be calculated
        :param min_ncap: Minimum accuracy to use for the computaton (start of the convergence check)
                         Must be >= nof_coefficients. If not so, is set automatically to nof_coefficients
        :param max_ncap: Maximum accuracy parameter. Forces exit if ncap reaches that value without convergence
                         Must be <= discretized_bath.max_nof_coefficients. If not so is set automatically to
        :param step_ncap: Number of star coefficients to be added in each step of the convergence
        :param stop_rel: Target relative deviation between successive steps. May be None if stop_abs is not None
        :param stop_abs: Target aboslute deviation between successive steps. May be None if stop_rel is not None
        :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                        sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                        possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                        specifically
        :param residual: If set True computes the residual for the tridiagonalization.
                         This may use extra memory!
        :param get_trafo: Returns the corresponding transformation matrix
        :return: info dict with entries: 'ncap': ncap used for the tridiagonalization.
                                         'res': Residual of the tridiagonalization
                                         'trafo': Transformation matrix if selected
        """
        if nof_coefficients > self.max_nof_coefficients:
            print('Error: Selected number of coefficients too high! Must increase max_nof_coefficients')
            raise AssertionError
        ncap = self.find_ncap(nof_coefficients, min_ncap=min_ncap, max_ncap=max_ncap, step_ncap=step_ncap,
                              stop_rel=stop_rel, stop_abs=stop_abs, permute=permute)
        return self.compute(nof_coefficients, ncap=ncap, permute=permute, residual=residual, get_trafo=get_trafo)

    def compute_from_stepwise_convergence(self, nof_coefficients, min_ncap=10, max_ncap=2000, step_ncap=1,
                                          stop_rel=1e-10, stop_abs=1e-10, permute=None, residual=False,
                                          get_trafo=False):
        """
            Central method, which computes the chain coefficients via tridigonalization from the discretized bath
            coefficients. Stores the result in the internal buffers of the object, accessible via
            c0, omega and t. Determines the accuracy parameter/number of discretized coefficients ncap for the
            computation using a convergence condition for the chain coefficients omega, t. Iteratively uses
            more coefficients from the discretized bath in every step.
            Convergence condition is applied successively. First only the first bath coefficients are calculated
            Then the second ones, then the third ones etc. In each of those superiterations only the latest
            bath coefficients are convergence testes and the previous ones are considered properly convergec.
            This method is generally faster than the from_convergence one
        :param nof_coefficients: Number of chain coefficients to be calculated
        :param min_ncap: Minimum accuracy to use for the computaton (start of the convergence check)
                         Must be >= nof_coefficients. If not so, is set automatically to nof_coefficients
        :param max_ncap: Maximum accuracy parameter. Forces exit if ncap reaches that value without convergence
                         Must be <= discretized_bath.max_nof_coefficients. If not so is set automatically to
        :param step_ncap: Number of star coefficients to be added in each step of the convergence
        :param stop_rel: Target relative deviation between successive steps. May be None if stop_abs is not None
        :param stop_abs: Target aboslute deviation between successive steps. May be None if stop_rel is not None
        :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                        sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                        possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                        specifically
        :param residual: If set True computes the residual for the tridiagonalization.
                         This may use extra memory!
        :param get_trafo: Returns the corresponding transformation matrix
        :return: info dict with entries: 'ncap': ncap used for the tridiagonalization.
                                         'res': Residual of the tridiagonalization
                                         'trafo': Transformation matrix if selected
        """

        if nof_coefficients > self.max_nof_coefficients:
            print('Error: Selected number of coefficients too high! Must increase max_nof_coefficients')
            raise AssertionError
        ncap = self.find_ncap_stepwise(nof_coefficients, min_ncap=min_ncap, max_ncap=max_ncap, step_ncap=step_ncap,
                                       stop_rel=stop_rel, stop_abs=stop_abs, permute=permute)
        return self.compute(nof_coefficients, ncap=ncap, permute=permute, residual=residual, get_trafo=get_trafo)

    def find_ncap(self, nof_coefficients, min_ncap=10, max_ncap=2000, step_ncap=1,
                  stop_rel=1e-10, stop_abs=1e-10, permute=None, threshold=1e-15):
        """
            Finds ncap, which satisfies the convergence condition outlined in compute_from_convergence.
            threshold marks the number, below which everything is treated as numerically 0 for the purpose of checking
            the convergence condition
        """
        assert stop_rel is not None or stop_abs is not None
        if self.discretized_bath.max_nof_coefficients < max_ncap:
            print('Buffer of discretized bath not sufficient to hold. Must fit at least max_ncap = ' + str(max_ncap) +
                  'elements. Set max_ncap to the maximum possible value of : ' +
                  str(self.discretized_bath.max_nof_coefficients))
            max_ncap = self.discretized_bath.max_nof_coefficients
        if stop_rel is None:
            stop_rel = np.inf
        if stop_abs is None:
            stop_abs = np.inf
        assert 1 <= nof_coefficients
        if min_ncap < nof_coefficients:
            print('min_ncap must be at least equal to the number of coefficients. Setting min_ncap=nof_coefficients')
            min_ncap = nof_coefficients
        self.sorting.select(permute)
        cutoff = self._get_cutoff(nof_coefficients)
        try:
            self._update_tridiag(min_ncap)
        except EOFCoefficients as err:
                print('min_ncap was set too high. Using maximum possible precision of ncap = ' +
                      str(err.nof_calc_coeff))
                return err.nof_calc_coeff
        # Init:
        alpha, beta, info = self.tridiag.get_tridiagonal(cutoff=cutoff, residual=False,
                                                         get_trafo=False)
        last_alpha = alpha
        last_beta = beta
        for curr_ncap in range(min_ncap + step_ncap, max_ncap + 1, step_ncap):
            # Update gamma/xi buffer with new coefficients:
            try:
                self._update_tridiag(curr_ncap)
            except EOFCoefficients as err:
                print('Limit for the calculation of gamma/xi coefficients reached. Using fixed max. accuracy of '
                      'ncap = ' + str(err.nof_calc_coeff))
                return err.nof_calc_coeff
            # Compute new omega/t
            alpha, beta, info = self.tridiag.get_tridiagonal(cutoff=cutoff, residual=False,
                                                             get_trafo=False)
            curr_alpha = alpha
            curr_beta = beta
            # Check global maximum difference
            if check_array_convergence(last_alpha, last_beta, curr_alpha, curr_beta, stop_abs, stop_rel,
                                       threshold=threshold):
                return curr_ncap
            # Update reference coefficients (taking views is important here):
            last_alpha = curr_alpha
            last_beta = curr_beta
        print('Maximum ncap reached without convergence. Using fixed maximum accuracy of ncap = ' + str(max_ncap))
        return max_ncap

    def find_ncap_stepwise(self, nof_coefficients, min_ncap=10, max_ncap=2000, step_ncap=1,
                           stop_rel=1e-10, stop_abs=1e-10, permute=None, threshold=1e-15):
        """
            Finds ncap, which satisfies the convergence condition outlined in compute_from_stepwise_convergence
            threshold marks the number, below which everything is treated as numerically 0 for the purpose of checking
            the convergence condition
        """
        assert stop_rel is not None or stop_abs is not None
        if self.discretized_bath.max_nof_coefficients < max_ncap:
            print('Buffer of discretized bath not sufficient to hold. Must fit at least max_ncap = ' + str(max_ncap) +
                  'elements. Set max_ncap to the maximum possible value of : ' +
                  str(self.discretized_bath.max_nof_coefficients))
            max_ncap = self.discretized_bath.max_nof_coefficients
        if stop_rel is None:
            stop_rel = np.inf
        if stop_abs is None:
            stop_abs = np.inf
        assert 1 <= nof_coefficients
        if min_ncap < nof_coefficients:
            print('min_ncap must be at least equal to the number of coefficients. Setting min_ncap=nof_coefficients')
            min_ncap = nof_coefficients
        self.sorting.select(permute)
        cutoff = self._get_cutoff(nof_coefficients)
        # Prepare gamma and xi buffers
        curr_ncap = min_ncap
        # Prepare gamma and xi buffers
        try:
            self._update_tridiag(min_ncap)
        except EOFCoefficients as err:
            print('min_ncap was set too high. Using maximum possible precision of ncap = ' + str(err.nof_calc_coeff))
            return err.nof_calc_coeff
        # Probe the required ncap which satisfy the stop_rel/stop_abs requirements for all coefficients
        for m in range(cutoff):
            # self.mapping.update_view(curr_ncap+1)
            # Initialize reference coefficients
            alpha, beta, info = self.tridiag.get_tridiagonal(cutoff=m+1, residual=False,
                                                             get_trafo=False)
            # Update reference coefficients
            last_alpha = alpha[m]
            last_beta = beta[m-1] if len(beta) > 0 else 0
            # Steadily increase ncap by step_ncap to see, when stop_rel/stop_abs requirements are fulfilled
            while True:
                next_ncap = curr_ncap + step_ncap
                # Check if we went above the ncap limit
                if next_ncap > max_ncap:
                    print('Maximum ncap reached without convergence. Using fixed maximum accuracy of ncap = ' +
                          str(curr_ncap))
                    return curr_ncap
                # Update gamma/xi buffer with new coefficients to next_ncap.
                try:
                    self._update_tridiag(next_ncap)
                except EOFCoefficients as err:
                    print('Limit for the calculation of gamma/xi coefficients reached. Using fixed max. accuracy of'
                          ' ncap = ' + str(err.nof_calc_coeff))
                    return err.nof_calc_coeff
                # Compute new omega/t
                alpha, beta, info = self.tridiag.get_tridiagonal(cutoff=m+1, residual=False,
                                                                 get_trafo=False)
                # Update reference coefficients
                curr_alpha = alpha[m]
                curr_beta = beta[m-1] if len(beta) > 0 else 0
                # Update curr_ncap for the next loop (which may be for the same m or for a new one),
                # this seems to work better in the mapping case, in contrast to the lanczos mapping, where
                # one should only update curr_ncap, if there there was no convergence in this step
                curr_ncap = next_ncap
                # Check if the next omega/t coefficients satisfy the stop_rel/stop_abs requirements
                if check_convergence(last_alpha, last_beta, curr_alpha, curr_beta, stop_abs, stop_rel,
                                     threshold=threshold):
                    if m == nof_coefficients-1:
                        return next_ncap
                    break
                # And update reference coefficients to the new ones which were just calculated
                last_alpha = curr_alpha
                last_beta = curr_beta
