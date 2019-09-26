from mapping.star.discretized_bath.asymmetric_bsdo import BSDODiscretizedAsymmetricBath


def get_bsdo(J, domain, nof_coefficients, orthpol_ncap=60000, mp_dps=30, force_sp=False, get_trafo=False,
             sort_by=None):
    """
        Returns bsdo star coefficients, see star.get_discretized_bath for an explanation of the arguments.
        Sort_by sorts the couplings and energies (if passed and not None), see utils.sorting.sort_star_coefficients
        for details on the parameter.
    :returns: gamma (couplings), xi (energies), info dict, which contains the transformation matrix if selected
    """
    bsdo_bath = BSDODiscretizedAsymmetricBath(J, domain, max_nof_coefficients=nof_coefficients,
                                              orthpol_ncap=orthpol_ncap, mp_dps=mp_dps, force_sp=force_sp,
                                              get_trafo=get_trafo)
    bsdo_bath.compute_all()
    gamma, xi = bsdo_bath.get_sorted(sort_by=sort_by)
    return gamma, xi, bsdo_bath.info
