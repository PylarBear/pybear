# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_bool import InBoolParamType







def _val_bool_param_value(
    _bool_param_key: str,
    _bool_param_value: InBoolParamType,
    _shrink_pass_can_be_None=False
) -> None:

    """
    Validate _bool_param_value.

    COMES IN AS
    list-like(
        list-like('grid_value1', etc.),
        None or integer > 0,
        'bool'
    )

    validate bool_params' dict value is a list-like that contains
    (i) a list-like of bool and/or None values
    (ii) a positive integer or None
    (iii) 'bool' (literal string 'bool')


    Parameters
    ----------
    _bool_param_key:
        str - the estimator parameter name to be grid-searched.
    _bool_param_value:
        InBoolParamType - the list-like of instructions for the multi-pass
        grid search of this boolean-valued parameter.
    _shrink_pass_can_be_None:
        bool, default=False - whether the second position is allowed to
        take the value None.


    Returns
    -------
    -
        None


    """


    _base_err_msg = f"bool param '{str(_bool_param_key)}' in :param: 'params' "


    # validate first position ** * ** * ** * ** * ** * ** * ** * ** * **
    # (i) a list of bool values
    _err_msg = (f"{_base_err_msg} --- "
        f"\nfirst position of the value must be a non-empty list-like that "
        f"\ncontains the first pass grid-search values."
    )

    if not all(map(
        isinstance,
        _bool_param_value[0],
        ((bool, type(None)) for _ in _bool_param_value[0])
    )):
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
        if _bool_param_value[1] is None:
            if _shrink_pass_can_be_None:
                raise UnicodeError
            else:
                raise Exception
        float(_bool_param_value[1])
        if isinstance(_bool_param_value[1], bool):
            raise Exception
        if int(_bool_param_value[1]) != _bool_param_value[1]:
            raise Exception
        if _bool_param_value[1] < 2:
            raise TimeoutError
    except UnicodeError:
        pass
    except TimeoutError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    del err_msg
    # END validate second position ** * ** * ** * ** * ** * ** * ** * **


    del _base_err_msg





