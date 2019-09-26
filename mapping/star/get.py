from mapping.star.get_discretized_bath import get_discretized_bath
from mapping.chain.get import get as get_chain, get_from_convergence, get_from_stepwise_convergence, from_bath, \
    from_bath_from_convergence, from_bath_from_stepwise_convergence
from mapping.utils.convert import convert_chain_to_star
from mapping.utils.sorting import sort_star_coefficients


def get(J, domain, nof_coefficients, disc_type='sp_quad', interval_type='lin', sort_by=None, **kwargs):
    """
        Returns star coefficients, see star.get_discretized_bath for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
        Note: In the case of bsdo discretization, there is no option to return the transformation matrix from chain
              to star. Use the star.bsdo.get function instead
    :returns: gamma (couplings), xi (energies)
    """
    disc_bath = get_discretized_bath(J, domain, max_nof_coefficients=nof_coefficients, disc_type=disc_type,
                                     interval_type=interval_type, **kwargs)
    disc_bath.compute_all()
    gamma, xi = disc_bath.get_sorted(sort_by=sort_by)
    return gamma, xi


def get_from_chain(J, domain, nof_coefficients, ncap=10000, disc_type='sp_quad', interval_type='lin',
                   mapping_type='lan_bath', permute=None, residual=True, low_memory=True, stable=False,
                   get_trafo=False, force_sp=False, mp_dps=30, sort_by=None, **kwargs):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization
        see chain.get and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = get_chain(J, domain, nof_coefficients, ncap=ncap, disc_type=disc_type,
                                   interval_type=interval_type, mapping_type=mapping_type, permute=permute,
                                   residual=residual, low_memory=low_memory, stable=stable,
                                   get_trafo=False, **kwargs)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)


def get_from_chain_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                               interval_type='log', mapping_type='sp_hes', permute=None, residual=True,
                               low_memory=True, stable=False, min_ncap=10, max_ncap=2000, step_ncap=1,
                               stop_rel=1e-10, stop_abs=1e-10, get_trafo=False, force_sp=False, mp_dps=30,
                               sort_by=None, **kwargs):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization
        see chain.get_from_convergence and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
        Defaults to logarithmic intervals for the direct discretization
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = get_from_convergence(J, domain, nof_coefficients, disc_type=disc_type,
                                              interval_type=interval_type, mapping_type=mapping_type,
                                              permute=permute, residual=residual,
                                              low_memory=low_memory, stable=stable, min_ncap=min_ncap,
                                              max_ncap=max_ncap, step_ncap=step_ncap,
                                              stop_rel=stop_rel, stop_abs=stop_abs, get_trafo=False, **kwargs)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)


def get_from_chain_stepwise_convergence(J, domain, nof_coefficients, disc_type='sp_quad',
                                        interval_type='log', mapping_type='sp_hes', permute=None,
                                        residual=True, low_memory=True, stable=False, min_ncap=10,
                                        max_ncap=2000, step_ncap=1, stop_rel=1e-10, stop_abs=1e-10,
                                        get_trafo=False, force_sp=False, mp_dps=30,
                                        sort_by=None, **kwargs):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization
        see chain.get_from_stepwise_convergence and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
        Defaults to logarithmic intervals for the direct discretization
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = get_from_stepwise_convergence(J, domain, nof_coefficients, disc_type=disc_type,
                                                       interval_type=interval_type, mapping_type=mapping_type,
                                                       permute=permute, residual=residual,
                                                       low_memory=low_memory, stable=stable, min_ncap=min_ncap,
                                                       max_ncap=max_ncap, step_ncap=step_ncap,
                                                       stop_rel=stop_rel, stop_abs=stop_abs, get_trafo=False, **kwargs)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)


def get_from_hr_star(discretized_bath, nof_coefficients, mapping_type='lan_bath', permute=None, residual=True,
                     low_memory=True, stable=False, get_trafo=False, force_sp=False, mp_dps=30, sort_by=None):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization using a pre-
        constructed discretized bath as 'high-resolution' star, which is mapped to chain, truncated and
        then mapped back,
        see chain.from_bath and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = from_bath(discretized_bath, nof_coefficients, mapping_type=mapping_type, permute=permute,
                                   residual=residual, low_memory=low_memory, stable=stable, get_trafo=get_trafo)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)


def get_from_hr_star_convergence(discretized_bath, nof_coefficients, mapping_type='sp_hes', permute=None, residual=True,
                                 low_memory=True, stable=False, min_ncap=10, max_ncap=2000, step_ncap=1, stop_rel=1e-10,
                                 stop_abs=1e-10, get_trafo=False, force_sp=False, mp_dps=30, sort_by=None):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization using a pre-
        constructed discretized bath as 'high-resolution' star, which is mapped to chain, truncated and
        then mapped back,
        see chain.from_bath_from_convergence and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = from_bath_from_convergence(discretized_bath, nof_coefficients, mapping_type=mapping_type,
                                                    permute=permute, residual=residual, low_memory=low_memory,
                                                    stable=stable, min_ncap=min_ncap, max_ncap=max_ncap,
                                                    step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                                    get_trafo=get_trafo)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)


def get_from_hr_star_stepwise_convergence(discretized_bath, nof_coefficients, mapping_type='sp_hes', permute=None,
                                          residual=True, low_memory=True, stable=False, min_ncap=10, max_ncap=2000,
                                          step_ncap=1, stop_rel=1e-10, stop_abs=1e-10, get_trafo=False, force_sp=False,
                                          mp_dps=30, sort_by=None):
    """
        Returns star coefficients, constructed from chain coefficients via diagonalization using a pre-
        constructed discretized bath as 'high-resolution' star, which is mapped to chain, truncated and
        then mapped back,
        see chain.from_bath_from_stepwise_convergence and convert_chain_to_star for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameters.
    :returns: gamma (couplings), xi (energies), info dict from both the conversion and the chain mapping
              if get_trafo is set True, the dict only contains the latest transformation (from chain to star here)
    """
    c0, omega, t, info = from_bath_from_stepwise_convergence(discretized_bath, nof_coefficients,
                                                             mapping_type=mapping_type,  permute=permute,
                                                             residual=residual, low_memory=low_memory,
                                                             stable=stable, min_ncap=min_ncap, max_ncap=max_ncap,
                                                             step_ncap=step_ncap, stop_rel=stop_rel, stop_abs=stop_abs,
                                                             get_trafo=get_trafo)
    gamma, xi, trafo_info = convert_chain_to_star(c0, omega, t, force_sp=force_sp, mp_dps=mp_dps, get_trafo=get_trafo)
    gamma, xi = sort_star_coefficients(gamma, xi, sort_by)
    return gamma, xi, info.update(trafo_info)
