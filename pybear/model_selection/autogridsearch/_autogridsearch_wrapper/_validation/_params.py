# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ._numerical_params import _numerical_param_value
from ._string_params import _string_param_value



def _params_validation(_params:[list, dict], total_passes:int) -> [list, dict]:

    """Validate numerical and string params

    Parameters
    ----------
    _params:
        A single dictionary or list of dictionaries. Each dictionary must
        contain parameters names as keys and lists that follow the format
        rules for string-type parameters and numerical parameters.

    Returns
    -------
    list -
        list of parameter dictionaries

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

    # total_passes must be int >= 1 ** * ** * ** * ** * ** * ** * ** * ** * **
    err_msg = f"'total_passes' must be an integer >= 1"
    try:
        float(total_passes)
        if isinstance(total_passes, bool):
            raise Exception
    except:
        raise TypeError(err_msg)

    if int(total_passes) != total_passes:
        raise TypeError(err_msg)

    total_passes = int(total_passes)

    if total_passes < 1:
        raise ValueError(err_msg)

    del err_msg
    # END total_passes must be int >= 1 ** * ** * ** * ** * ** * ** * ** * ** *


    err_msg = f"'_params' must be a dictionary or a list-like of dictionaries"

    try:
        iter(_params)
    except:
        raise TypeError(err_msg)

    if isinstance(_params, dict):
        _params = [_params]
    else:
        for item in _params:
            if not isinstance(item, dict):
                raise TypeError(err_msg)

    del err_msg

    # _params must be a list of dictionaries at this point

    for _idx, _param_dict in enumerate(_params):

        for _key in _param_dict:

            # keys are strings
            if not isinstance(_key, str):
                raise TypeError(
                    f"param dict {_idx+1} -- parameter keys must be strings "
                    f"corresponding to args/kwargs of an estimator"
            )

            # values are list-like
            try:
                iter(_param_dict[_key])
                if isinstance(_param_dict[_key], (dict, set, str)):
                    raise Exception
            except:
                raise TypeError(f"param dict {_idx+1} parameter values must be "
                                f"list-like")




            # last posn of value must be a string of dtype, search type

            allowed = ['string', 'hard_float', 'hard_integer', 'soft_float',
                       'soft_integer', 'fixed_float', 'fixed_integer']

            err_msg = (f"param dict {_idx+1}, {_key} -- last position must be a "
                       f"string in [{', '.join(allowed)}]")
            try:
                _params[_idx][_key][-1] = _param_dict[_key][-1].lower()
            except AttributeError:
                raise TypeError(err_msg)
            except Exception as e:
                raise Exception(f"param dict {_idx+1}, {_key} -- dtype/search "
                                f"string failed for uncontrolled reason -- {e}")

            if _param_dict[_key][-1] not in allowed:
                raise ValueError(err_msg)

            del err_msg

            if _param_dict[_key][-1] == 'string':
                _params[_idx][_key] = \
                    _string_param_value(_key, _param_dict[_key])
            else:
                _params[_idx][_key] = \
                    _numerical_param_value(_key, _param_dict[_key], total_passes)


    return _params









