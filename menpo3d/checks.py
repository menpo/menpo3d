import collections
import numpy as np


def check_max_iters(max_iters, n_scales):
    r"""
    Function that checks the value of a `max_iters` parameter defined for
    multiple scales. It must be `int` or `list` of `int`.

    Parameters
    ----------
    max_iters : `int` or `list` of `int`
        The value to check.
    n_scales : `int`
        The number of scales.

    Returns
    -------
    max_iters : `list` of `int`
        The list of values per scale.

    Raises
    ------
    ValueError
        max_iters can be integer, integer list containing 1 or {n_scales}
        elements or None
    """
    if type(max_iters) is int:
        max_iters = [np.round(max_iters/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) == 1 and n_scales > 1:
        max_iters = [np.round(max_iters[0]/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) != n_scales:
        raise ValueError('max_iters can be integer, integer list '
                         'containing 1 or {} elements or '
                         'None'.format(n_scales))
    return np.require(max_iters, dtype=np.int)


def check_multi_scale_param(n_scales, types, param_name, param):
    r"""
    General function for checking a parameter defined for multiple scales. It
    raises an error if the parameter is not an iterable with the correct size and
    correct types.

    Parameters
    ----------
    n_scales : `int`
        The number of scales.
    types : `tuple`
        The `tuple` of variable types that the parameter is allowed to have.
    param_name : `str`
        The name of the parameter.
    param : `types`
        The parameter value.

    Returns
    -------
    param : `list` of `types`
        The list of values per scale.

    Raises
    ------
    ValueError
        {param_name} must be in {types} or a list/tuple of {types} with the same
        length as the number of scales
    """
    error_msg = "{0} must be in {1} or a list/tuple of " \
                "{1} with the same length as the number " \
                "of scales".format(param_name, types)

    # Could be a single value - or we have an error
    if isinstance(param, types):
        return [param] * n_scales
    elif not isinstance(param, collections.Iterable):
        raise ValueError(error_msg)

    # Must be an iterable object
    len_param = len(param)
    isinstance_all_in_param = all(isinstance(p, types) for p in param)

    if len_param == 1 and isinstance_all_in_param:
        return list(param) * n_scales
    elif len_param == n_scales and isinstance_all_in_param:
        return list(param)
    else:
        raise ValueError(error_msg)


def check_parameters(parameters, n_active_components):
    params = parameters
    if params is None:
        params = np.zeros(n_active_components)
    elif len(params) > n_active_components:
        params = np.asarray(params[:n_active_components])
    elif len(params) < n_active_components:
        tmp = params
        params = np.zeros(n_active_components)
        params[:len(tmp)] = np.asarray(tmp)
    return params
