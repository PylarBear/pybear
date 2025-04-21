# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_str import InStrParamType



def _val_string_param_value(
    _string_param_key: str,
    _string_param_value: InStrParamType,
    _shrink_pass_can_be_None=False
) -> None:

    """
    Validate _string_param_value.

    COMES IN AS
    list-like(
        list-like('grid_value1', 'grid_value2', etc.),
        None or integer > 0,
        'string'
    )

    validate string_params' dict value is a list-like that contains
    (i) a list-like of str/None values
    (ii) a positive integer or None
    (iii) 'string' (literal string 'string')


    Parameters
    ----------
    _string_param_key:
        str - the estimator parameter name to be grid-searched.
    _string_param_value:
        InStrParamType - the list-like of instructions for the multi-pass
        grid search of this string-values parameter.
    _shrink_pass_can_be_None:
        bool, default=False - whether the second position is allowed to
        take the value None.


    Returns
    -------
    -
        None


    """


    _base_err_msg = f"string param '{str(_string_param_key)}' in :param: 'params' "

    if not isinstance(_string_param_key, str):
        raise TypeError(f"{_base_err_msg} --- param key must be a string")

    # validate outer container object ** * ** * ** * ** * ** * ** * ** *
    try:
        iter(_string_param_value)
        if isinstance(_string_param_value, (dict, str, set)):
            raise Exception
        if len(_string_param_value) != 3:
            raise UnicodeError
    except UnicodeError:
        raise ValueError(_base_err_msg + "--- value must contain 3 things: "
            "\nfirst search grid, \nshrink pass, \nthe literal string 'string'")
    except Exception as e:
        raise TypeError(_base_err_msg + f"--- value must be list-like")
    # END validate outer container object ** * ** * ** * ** * ** * ** *

    # validate first position ** * ** * ** * ** * ** * ** * ** * ** * **
    # (i) a list of str values
    _err_msg = (f"{_base_err_msg} --- "
        f"\nfirst position of the value must be a non-empty list-like that "
        f"\ncontains the first pass grid-search values (either strings or None)."
    )

    try:
        iter(_string_param_value[0])
        if isinstance(_string_param_value[0], (dict, str)):
            raise Exception
        if len(_string_param_value[0]) == 0:
            raise UnicodeError
        if not all(map(
            isinstance,
            _string_param_value[0],
            ((str, type(None)) for _ in _string_param_value[0])
        )):
            raise Exception
    except UnicodeError:
        raise ValueError(_err_msg)
    except Exception as e:
        raise TypeError(_err_msg)

    del _err_msg
    # END validate first position ** * ** * ** * ** * ** * ** * ** * **

    # validate second position ** * ** * ** * ** * ** * ** * ** * ** * *
    err_msg = (f"{_base_err_msg} --- "
        f"\nsecond position of the value must be None or an integer > 1 "
        f"\nindicating the pass on which to reduce a param's grid to "
        f"\nonly a single search value."
    )

    try:
        if _string_param_value[1] is None:
            if _shrink_pass_can_be_None:
                raise UnicodeError
            else:
                raise Exception
        float(_string_param_value[1])
        if isinstance(_string_param_value[1], bool):
            raise Exception
        if int(_string_param_value[1]) != _string_param_value[1]:
            raise Exception
        if _string_param_value[1] < 2:
            raise TimeoutError
    except UnicodeError:
        pass
    except TimeoutError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    del err_msg
    # END validate second position ** * ** * ** * ** * ** * ** * ** * **

    # validate third position ** * ** * ** * ** * ** * ** * ** * ** * **
    err_msg = (f"{_base_err_msg}' --- "
        f"\nthird position of the value must be the literal string 'string'"
    )

    if not isinstance(_string_param_value[2], str):
        raise TypeError(err_msg)

    if _string_param_value[2].lower() != 'string':
        raise ValueError(err_msg)

    del err_msg
    # END validate third position ** * ** * ** * ** * ** * ** * ** * **

    del _base_err_msg





