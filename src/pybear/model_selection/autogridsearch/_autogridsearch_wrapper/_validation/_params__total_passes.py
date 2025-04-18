# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from ._params_numerical import _numerical_param_value
from ._params_string import _string_param_value
from ._params_bool import _bool_param_value

from .._type_aliases import ParamsType



def _params__total_passes(
    _params: ParamsType,
    _total_passes: int
) -> tuple[ParamsType, int]:

    """
    Validate numerical, string, and bool params within _params, and
    standardize the format of _params, vis-à-vis total_passes.


    Parameters
    ----------
    _params:
        dict[str, Sequence[Sequence, Sequence|int], str] - A single
        dictionary that contains parameter names as keys and lists that
        follow the format rules for string, bool, and numerical
        parameters. AutoGridSearch does not accept lists of multiple
        params dictionaries in the same way that Scikit-Learn and Dask
        accept multiple param_grids.

    _total_passes:
        int - the number of grid searches to perform. The actual number
        of passes run can be different from this number based on the
        setting for the total_passes_is_hard argument. If total_passes_is_hard
        is True, then the maximum number of total passes will always be
        the value assigned to total_passes. If total_passes_is_hard is
        False, a round that performs a 'shift' operation will increment
        the total number of passes, essentially causing shift passes to
        not count toward the total number of passes. Read elsewhere in
        the docs for more information about 'shifting' and 'drilling'.

    Returns
    -------
    -
        _params: dict - dictionary of grid-building instructions for all
        parameters.
        _total_passes: int - the total number of GridSearches to perform.

    Examples
    --------
    string parameter:
        {'solver': [['saga', 'lbfgs'], 3, 'string']
    numerical parameter:
        {'C': [[10, 20, 30], [3,3,3], 'soft_float']}
        -- or --
        {'C': ['linspace', 10, 30, [3,3,3], 'soft_float']}

    a full parameter dictionary:
        {
        'C': ['logspace', -5, 5, [11, 11, 11], 'soft_float'],
        'l1_ratio': ['linspace', 0, 1, 21, 'fixed_float'],
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


    # validate this even though it may not be needed
    err_msg = f"'total_passes' must be an integer >= 1"
    try:
        float(_total_passes)
        if isinstance(_total_passes, bool):
            raise Exception
    except:
        raise TypeError(err_msg)

    if int(_total_passes) != _total_passes:
        raise TypeError(err_msg)

    _total_passes = int(_total_passes)

    if _total_passes < 1:
        raise ValueError(err_msg)

    del err_msg
    # END total_passes must be int >= 1 ** * ** * ** * ** * ** * ** * ** * ** *



    # params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if not isinstance(_params, dict):
        raise TypeError(f"'_params' must be a dictionary")


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
        # points can be int or list-like that is not a set.
        _points = _params[_param][-2]
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
    if len(_unq_points) == 0:
        # there were no list-likes for points, so use the total_passes kwarg
        pass
    elif len(_unq_points) > 1:
        raise ValueError(f"when 'points' are passed to parameters as lists, "
            f"all points lists must be the same length, the total number "
            f"of passes")
    elif len(_unq_points) == 1:
        # must be an integer, because it came from len()
        _total_passes = int(_unq_points[0])

    del _unq_points

    # END finalize total_passes ** * ** * ** * ** * ** * ** * ** * ** *

    for _key in _params:

        # keys are strings
        if not isinstance(_key, str):
            raise TypeError(
                f"parameter keys must be strings corresponding to "
                f"args/kwargs of an estimator"
        )

        # values are list-like
        try:
            iter(_params[_key])
            if isinstance(_params[_key], (dict, set, str)):
                raise Exception
        except:
            raise TypeError(f"parameter values must be list-like")


        # last posn of value must be a string of dtype / search type

        allowed = ['string', 'hard_float', 'hard_integer', 'soft_float',
                   'soft_integer', 'fixed_float', 'fixed_integer', 'bool']

        err_msg = (f"{_key} -- last position must be a string in \n"
                   f"[{', '.join(allowed)}]")
        try:
            _params[_key][-1] = _params[_key][-1].lower()
        except AttributeError:
            raise TypeError(err_msg)
        except Exception as e:
            raise Exception(f"{_key} -- dtype/search string failed for "
                            f"uncontrolled reason -- {e}")

        if _params[_key][-1] not in allowed:
            raise ValueError(err_msg)

        del err_msg

        if _params[_key][-1] == 'string':
            _params[_key] = _string_param_value(_key, _params[_key])
        elif _params[_key][-1] == 'bool':
            _params[_key] = _bool_param_value(_key, _params[_key])
        else:
            _params[_key] = _numerical_param_value(
                _key, _params[_key], _total_passes
            )
    # END params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **




    return _params, _total_passes































