# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import numpy as np

from ._params_numerical import _cond_numerical_param_value
from ._params_string import _cond_string_param_value
from ._params_bool import _cond_bool_param_value

from .._type_aliases import (
    InParamsType,
    ParamsType
)



def _cond_params__total_passes(
    _params: InParamsType,
    _total_passes: numbers.Integral,
    _inf_shrink_pass: numbers.Integral
) -> tuple[ParamsType, int]:

    """
    Standardize the format of _params, vis-à-vis total_passes.


    Parameters
    ----------
    _params:
        InParamsType - A single dictionary that contains parameter names
        as keys and list-likes that follow the format rules for string,
        bool, and numerical parameters as values. AutoGridSearch does
        not accept lists of multiple params dictionaries in the same way
        that Scikit-Learn and Dask accept multiple param_grids.
    _total_passes:
        numbers.Integral - the number of grid searches to perform. The
        actual number of passes run can be different from this number
        based on the setting for :param: `total_passes_is_hard`. If
        `total_passes_is_hard` is True, then the maximum number of total
        passes will always be the value assigned to `total_passes`. If
        `total_passes_is_hard` is False, a round that performs a 'shift'
        operation will increment the total number of passes, essentially
        causing shift passes to not count toward the total number of
        passes. Read elsewhere in the docs for more information about
        'shifting' and 'drilling'.
    _inf_shrink_pass:
        numbers.Integral - the larger integer to substitute in the shrink
        pass  position if the user passed None for shrink pass


    Returns
    -------
    -
        _params: ParamsType - dictionary of grid-building instructions
        for all parameters.
        _total_passes: int - the total number of GridSearches to perform.


    Examples
    --------
    string parameter:
        {'solver': [['saga', 'lbfgs'], 3, 'string']
    bool parameter:
        {'remove_empty_rows': [[True, False], 3, 'bool']
    numerical parameter:
        {'C': [[10, 20, 30], [3,3,3], 'soft_float']}
    a full parameter dictionary:
        {
            'C': [np.logspace(-5, 5, 11), [11, 11, 11], 'soft_float'],
            'l1_ratio': [np.linspace(0, 1, 21), [21, 6, 6], 'fixed_float'],
            'solver': [['saga', 'lbfgs'], 2, 'string']
        }


    """

    # _total_passes must be int >= 1 ** * ** * ** * ** * ** * ** * ** * ** * **

    # this number may not be needed if _params contains 'points' that are
    # list-type (where the length of the list of points is the number of
    # passes.) If points are passed as lists to multiple parameters, the
    # lengths must all be equal. String and bool params do not take points
    # internally; this must be set with the total_passes arg or inferred
    # from other params that have list-like 'points'. Numerical params
    # can take a list-type or a single integer for 'points'. If no params
    # are passed with a list-type for points, or all string / bool
    # parameters are passed, the total_passes arg is used.

    _total_passes = int(_total_passes)

    # END total_passes must be int >= 1 ** * ** * ** * ** * ** * ** * ** * ** *

    # params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # must get a finalized number for total_passes before looping over
    # _params because _{string/numerical/bool}_param_value standardize
    # the 'points' slot into a list with len == total_passes. Validation
    # of the 'points' slot is handled in those 3 modules and that's where
    # it will stay to avoid another huge surgery to the _numerical_param_value
    # module. So without having validated the 'points' slot here, look
    # for list-types in the [-2] slot and if there are list-types, use
    # that to infer total_passes and override the value for the
    # total_passes kwarg.

    _POINTS_LENS = []
    for _param in _params:
        # points can be int or list-like[int].
        _points = _params[_param][1]
        try:
            iter(_points)
            if isinstance(_points, (str, dict, set)):
                continue  # except on this later in _string _bool or _numerical
            _POINTS_LENS.append(len(_points))
        except:
            continue

        del _points

    _unq_points = np.unique(_POINTS_LENS)
    del _POINTS_LENS

    if len(_unq_points) == 1:
        # must be an integer, because it came from len()
        _total_passes = int(_unq_points[0])

    del _unq_points

    # END finalize total_passes ** * ** * ** * ** * ** * ** * ** * ** *

    for _key in _params:

        _params[_key] = list(_params[_key])

        # pizza this is probably redundant
        _params[_key][-1] = _params[_key][-1].lower()

        if _params[_key][-1] == 'string':
            _params[_key] = _cond_string_param_value(_params[_key], _inf_shrink_pass)
        elif _params[_key][-1] == 'bool':
            _params[_key] = _cond_bool_param_value(_params[_key], _inf_shrink_pass)
        else:
            _params[_key] = _cond_numerical_param_value(_params[_key], _total_passes)
    # END params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    return _params, _total_passes































