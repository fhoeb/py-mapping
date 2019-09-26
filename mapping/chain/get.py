"""
    Main interface for the generation of chain coefficients
"""
from mapping.chain.get_mapping import get_mapping
from mapping.chain.bsdo import get_bsdo, get_bsdo_from_convergence
from mapping.star.get_discretized_bath import get_discretized_bath


def get(J, domain, nof_coefficients, ncap=None, disc_type='sp_quad', interval_type='lin',
        mapping_type='lan_bath', permute=None, residual=True, low_memory=True, stable=False, get_trafo=False,
        **kwargs):
    """
        Main interface for the generation of chain coefficients.
    :param J: Spectral density of the bath as vectorized python function of one argument
    :param domain: Domain (support) of the spectral density as list/tuple/numpy array like [a, b]
    :param nof_coefficients: Desired number of chain coefficients (=number of bath sites)
    :param ncap: Accuracy parameter for the calculation of the coefficients (=orthpol_ncap for bsdo,
                 otherwise represents the number of coefficients in the discretized star to be used for the mapping)
                 Default is None, which sets ncap=nof_coefficients for non bsdo (unitary map) and ncap=60000 for bsdo.
                 If set < nof_coefficient, it is automatically set equal to nof_coefficients.
    :param disc_type: Type of the discretization (see star.get_discretized_bath for an in-depth explanation)
    :param interval_type: Type of intervals for the discretization (see star.get_discretized_bath
                          for an in-depth explanation)
    :param mapping_type: Type of the mapping to be used (see chain.get_mapping for an in-depth explanation)
    :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                    sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                    possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                    specifically
    :param residual: If set True computes the residual for the tridiagonalization.
                     This may use extra memory, specifically if low_memory was set True.
    :param low_memory: Uses a more memory efficient version of the lanczos algorithms, which may be slightly slower
    :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                   some stability problems encountered with Lanczos tridiagonalization
    :param get_trafo: Returns the corresponding transformation matrix
    :param kwargs: Additional parameters for the discretization
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings),
            info dict with keys: 'ncap': Number of star discretization points/orthpol_ncap,
                                 'res': Computed residual of the tridiagonaliation
                                 'trafo': Transformation matix between star and chain
    """
    if disc_type == 'bsdo':
        # mapping back and forth would be superfluous, just use orthpol directly
        c0, omega, t = get_bsdo(J, domain, nof_coefficients=nof_coefficients,
                                orthpol_ncap=ncap)
        info = dict()
        info['ncap'] = ncap
        info['res'] = None
        info['trafo'] = None
    else:
        if ncap is None:
            ncap = nof_coefficients
        disc_bath = get_discretized_bath(J, domain, max_nof_coefficients=max(ncap, nof_coefficients),
                                         disc_type=disc_type,
                                         interval_type=interval_type, **kwargs)
        disc_bath.compute_all()
        mapping = get_mapping(disc_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                              low_memory=low_memory, stable=stable)
        info = mapping.compute(nof_coefficients, ncap=ncap, permute=permute, residual=residual, get_trafo=get_trafo)
        c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info


def from_bath(discretized_bath, nof_coefficients, mapping_type='lan_bath', permute=None, residual=True,
              low_memory=True, stable=False, ncap=None, get_trafo=False):
    """
        Interface for the generation of chain coefficients from pre-constructed discretized_bath objects.
    :param discretized_bath: Discretized bath object
    :param nof_coefficients: Desired number of chain coefficients (=number of bath sites)
    :param mapping_type: Type of the mapping to be used (see chain.get_mapping for an in-depth explanation)
    :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                    sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                    possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                    specifically
    :param residual: If set True computes the residual for the tridiagonalization.
                     This may use extra memory, specifically if low_memory was set True.
    :param low_memory: Uses a more memory efficient version of the lanczos algorithms, which may be slightly slower
    :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                   some stability problems encountered with Lanczos tridiagonalization
    :param ncap: Accuracy parameter (number of discretized coefficients). Default is None. If None,
                 we set ncap=nof_coefficients (unitary map)
    :param get_trafo: Returns the corresponding transformation matrix
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings),
            info dict with keys: 'ncap': Number of star discretization points/orthpol_ncap,
                                 'res': Computed residual of the tridiagonaliation
                                 'trafo': Transformation matix between star and chain
    """
    if ncap is None:
        ncap = nof_coefficients
    mapping = get_mapping(discretized_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                          low_memory=low_memory, stable=stable)
    info = mapping.compute(ncap, permute=permute, residual=residual, get_trafo=get_trafo)
    c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info


def get_from_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                         interval_type='log', mapping_type='sp_hes', permute=None, residual=True,
                         low_memory=True, stable=False, min_ncap=10, max_ncap=2000, step_ncap=1,
                         stop_rel=1e-10, stop_abs=1e-10, get_trafo=False, **kwargs):
    """
        Interface for the generation of chain coefficients, which uses a convergence condition.
        See BaseMapping.compute_from_convergence for details.
        Defaults to logarithmic intervals for the direct discretization
    :param J: Spectral density of the bath as vectorized python function of one argument
    :param domain: Domain (support) of the spectral density as list/tuple/numpy array like [a, b]
    :param nof_coefficients: Desired number of chain coefficients (=number of bath sites)
    :param min_ncap: Minimum accuracy to use for the computaton (start of the convergence check)
                     Must be >= nof_coefficients. If not so, is set automatically to nof_coefficients
    :param max_ncap: Maximum accuracy parameter. Forces exit if ncap reaches that value without convergence
                     Must be <= discretized_bath.max_nof_coefficients. If not so is set automatically to
    :param step_ncap: Number of star coefficients to be added in each step of the convergence
    :param stop_rel: Target relative deviation between successive steps. May be None if stop_abs is not None
    :param stop_abs: Target aboslute deviation between successive steps. May be None if stop_rel is not None
    :param disc_type: Type of the discretization (see star.get_discretized_bath for an in-depth explanation)
    :param interval_type: Type of intervals for the discretization (see star.get_discretized_bath
                          for an in-depth explanation)
    :param mapping_type: Type of the mapping to be used (see chain.get_mapping for an in-depth explanation)
    :param permute: If the star coefficients should be permuted before each tridiagonalization (essentially
                    sorting them, see utils.sorting.sort_star_coefficients for an explanation of the
                    possible parameters). This may help increase numerical stability for Lanczos tridiagonalization
                    specifically
    :param residual: If set True computes the residual for the tridiagonalization.
                     This may use extra memory, specifically if low_memory was set True.
    :param low_memory: Uses a more memory efficient version of the lanczos algorithms, which may be slightly slower
    :param stable: Uses a stable summation algorithm, which is much slower but may help counteract some
                   some stability problems encountered with Lanczos tridiagonalization
    :param get_trafo: Returns the corresponding transformation matrix
    :param kwargs: Additional parameters for the discretization
    :return: c0 (System-Bath coupling), omega (Bath energies), t (bath-bath couplings),
            info dict with keys: 'ncap': Number of star discretization points/orthpol_ncap,
                                 'res': Computed residual of the tridiagonaliation
                                 'trafo': Transformation matix between star and chain
    """
    if disc_type == 'bsdo':
        # mapping back and forth would be superfluous, just use orthpol directly
        c0, omega, t, info = get_bsdo_from_convergence(J, domain, nof_coefficients,
                                                       min_ncap=min_ncap, step_ncap=step_ncap,
                                                       max_ncap=max_ncap, stop_abs=stop_abs,
                                                       stop_rel=stop_rel)
        info['trafo'] = None
        info['res'] = None
    else:
        disc_bath = get_discretized_bath(J, domain, max_nof_coefficients=max(max_ncap, nof_coefficients),
                                         disc_type=disc_type,
                                         interval_type=interval_type, **kwargs)
        mapping = get_mapping(disc_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                              low_memory=low_memory, stable=stable)
        info = mapping.compute_from_convergence(nof_coefficients, min_ncap=min_ncap, max_ncap=max_ncap,
                                                step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                                permute=permute, residual=residual, get_trafo=get_trafo)
        c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info


def from_bath_from_convergence(discretized_bath, nof_coefficients, mapping_type='sp_hes', permute=None, residual=True,
                               low_memory=True, stable=False, min_ncap=10, max_ncap=2000, step_ncap=1,
                               stop_rel=1e-10, stop_abs=1e-10, get_trafo=False):
    """
        Uses the same convergence condition for the discretized coefficients as get_from_discretization.
        Parameters are a mix between the ones from get_from_discretization and from_bath
    """
    mapping = get_mapping(discretized_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                          low_memory=low_memory, stable=stable)
    info = mapping.compute_from_convergence(nof_coefficients, min_ncap=min_ncap,
                                            max_ncap=max(max_ncap, nof_coefficients),
                                            step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                            permute=permute, residual=residual, get_trafo=get_trafo)
    c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info


def get_from_stepwise_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                                  interval_type='log', mapping_type='sp_hes', permute=None,
                                  residual=True, low_memory=True, stable=False, min_ncap=10,
                                  max_ncap=2000, step_ncap=1, stop_rel=1e-10, stop_abs=1e-10,
                                  get_trafo=False, **kwargs):
    """
        Parameters are the same as for get_from_convergence. Uses a slightly different convergence condition
        (see BaseMapping.compute_from_stepwise_convergence for an explanation), which is often
        faster than the other one.
        Defaults to logarithmic intervals for the direct discretization
    """
    if disc_type == 'bsdo':
        # mapping back and forth would be superfluous, just use orthpol directly
        # There is no bsdo stepwise convergence implemented currently so just take the normal one
        c0, omega, t, info = get_bsdo_from_convergence(J, domain, nof_coefficients,
                                                       min_ncap=min_ncap, step_ncap=step_ncap,
                                                       max_ncap=max_ncap, stop_abs=stop_abs,
                                                       stop_rel=stop_rel)
        info['trafo'] = None
        info['res'] = None
    else:
        disc_bath = get_discretized_bath(J, domain, max_nof_coefficients=max(max_ncap, nof_coefficients),
                                         disc_type=disc_type,
                                         interval_type=interval_type, **kwargs)
        mapping = get_mapping(disc_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                              low_memory=low_memory, stable=stable)
        info = mapping.compute_from_stepwise_convergence(nof_coefficients, min_ncap=min_ncap, max_ncap=max_ncap,
                                                         step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                                         permute=permute, residual=residual, get_trafo=get_trafo)
        c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info


def from_bath_from_stepwise_convergence(discretized_bath, nof_coefficients, mapping_type='sp_hes', permute=None,
                                        residual=True, low_memory=True, stable=False, min_ncap=10, max_ncap=2000,
                                        step_ncap=1, stop_rel=1e-10, stop_abs=1e-10, get_trafo=False):
    """
        Uses the same convergence condition for the discretized coefficients as get_from_stepwise_discretization.
        Parameters are a mix between the ones from get_from_stepwise_discretization and from_bath
    """
    mapping = get_mapping(discretized_bath, mapping_type=mapping_type, max_nof_coefficients=nof_coefficients,
                          low_memory=low_memory, stable=stable)
    info = mapping.compute_from_stepwise_convergence(nof_coefficients, min_ncap=min_ncap,
                                                     max_ncap=max(max_ncap, nof_coefficients),
                                                     step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                                     permute=permute, residual=residual, get_trafo=get_trafo)
    c0, omega, t = mapping.get_coefficients()
    return c0, omega, t, info
