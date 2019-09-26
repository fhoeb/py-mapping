import numpy as np


def get_asymmetric_interval_points(domain, nof_intervals, interval_type='lin', get_spacing=True, **kwargs):
    """
        Returns the endpoints (and optionally also the midpoints) of the intervals for direct discretization
    :param domain: List/tuple of two elements for the left and right boundary of the domain of the spectral density
    :param nof_intervals: Number of desired intervals for the discretization (=nof_points-1)
    :param interval_type: see star.get_discretized_bath for an explanation of the available types
    :param get_spacing: Returns the length of the intervals along with the points
    :param kwargs: Allowed are: 'Lambda': Parameter for the logarithmic discretization
                                'custom_fcn': If the interval_type was selected to be custom_fcn.
                                              star.get_discretized_bath for details
                                'custom_arr': If the interval_type was selected to be custom_arr.
                                              star.get_discretized_bath for details
    :returns: numpy array of the interval grid points (length=nof_intervals+1),
              length of the intervals´(as float if uniform, as numpy array otherwise) if get_spacing was set True
    """
    if interval_type == 'lin':
        points, dx = np.linspace(domain[0], domain[1], num=nof_intervals+1, retstep=True)
    elif interval_type == 'log':
        if domain[0] < 0 < domain[1]:
            print('Cannot use non symmetric logarithmic discretization with an interval that contains 0')
            raise AssertionError
        try:
            Lambda = float(kwargs['Lambda'])
        except KeyError:
            Lambda = 1.1
        points = (domain[0] + (domain[1] - domain[0]) * np.power(Lambda, -np.arange(nof_intervals+1)))[::-1]
        dx = (points[1:] - points[:-1])
    elif interval_type == 'custom_fcn':
        try:
            fcn = kwargs['custom_fcn']
        except KeyError:
            print('Warning: Keyword custom_fcn not detected. Using linear discretization')
            return get_asymmetric_interval_points(domain, nof_intervals, interval_type='lin', get_spacing=get_spacing,
                                                  **kwargs)
        v_fcn = np.vectorize(fcn)
        points = v_fcn(np.arange(0, nof_intervals+1))
        # sort coefficients in ascending order
        points = np.sort(points, kind='mergesort')
        dx = (points[1:] - points[:-1])
    elif interval_type == 'custom_arr':
        try:
            points = kwargs['custom_arr']
        except KeyError:
            print('Warning: Keyword custom_arr not detected. Using linear discretization')
            return get_asymmetric_interval_points(domain, nof_intervals, interval_type='lin', get_spacing=get_spacing,
                                                  **kwargs)
        if len(points) < nof_intervals:
            print('Custom point array must be at least ' + nof_intervals + ' elements large')
            raise AssertionError
        points = points[:nof_intervals]
        # sort coefficients in ascending order
        points = np.sort(points, kind='mergesort')
        dx = (points[1:] - points[:-1])
    else:
        print('Unsupported interval type')
        raise AssertionError
    if get_spacing:
        return points, dx
    else:
        return points


def get_symmetric_interval_points(domain, nof_intervals, interval_type='lin', get_spacing=True, **kwargs):
    """
        See get_symmetric_interval_points. Except custom_arr and custom_fcn are not supported here and
        the function returns as follows:
    :returns: numpy array of the interval grid points (length=nof_intervals+1) of the positive part of the domain,
              numpy array of the interval grid points (length=nof_intervals+1) of the negative part of the domain,
              length of the positive intervals, length of the negative intervals
              (as float if uniform, as numpy array otherwise) if get_spacing was set True for both
    """
    if interval_type == 'lin':
        points_p, dx_p = np.linspace(0, domain[1], num=nof_intervals + 1, retstep=True)
        points_m, dx_m = np.linspace(domain[0], 0, num=nof_intervals + 1, retstep=True)
    elif interval_type == 'log':
        try:
            Lambda = float(kwargs['Lambda'])
        except KeyError:
            Lambda = 1.1
        points_p = (domain[1] * np.power(Lambda, -np.arange(nof_intervals + 1)))[::-1]
        points_m = domain[0] * np.power(Lambda, -np.arange(nof_intervals + 1))
        dx_p, dx_m = (points_p[1:] - points_p[:-1]), (points_m[1:] - points_m[:-1])
    else:
        print('Unsupported interval type')
        raise AssertionError
    if get_spacing:
        return points_p, points_m, dx_p, dx_m
    else:
        return points_p, points_m

