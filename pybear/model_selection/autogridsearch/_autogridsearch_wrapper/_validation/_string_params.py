# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _string_param_value(_string_param_key:str, _string_param_value) -> list:

    """
    Validate _string_param_value --- standardize format

    COMES IN AS
    list-like(
                list-like('grid_value1', 'grid_value2', etc.),
                None or integer > 0,
                'string'
    )

    validate string_params' dict value is a list-like that contains
    (i) a list-like of str/None values
    (ii) a positive integer or None
    (iii) 'string' (a string-type of the word string)

    GOES OUT AS
    [
    ['grid_value1', 'grid_value2', etc.],
    1_000_000 or integer > 0,
    'string'
    ]


    """

    if not isinstance(_string_param_key, str):
        raise TypeError(f"_string_param_key must be a string")


    err_msg = (f"string_param {_string_param_key} -- values must be list-like "
               f"and contain these 3 items: "
               f"\n1) a list-like holding the grid-search values; "
               f"\ncannot be empty and must contain the values (either strings "
               f"or None-type) for its respective arg/kwarg of the estimator "
               f"\n2) None or an integer > 0 indicating the autogridsearch pass "
               f"on which to reduce this param's grid to only a single value "
               f"\n3) a string-like that says the word 'string'")

    # validate container object ** * ** * ** * ** * ** * ** * ** * ** *
    try:
        iter(_string_param_value)
    except:
        raise TypeError(err_msg)

    if isinstance(_string_param_value, (set, dict, str)):
        raise TypeError(err_msg)

    _string_param_value = list(_string_param_value)

    if len(_string_param_value) != 3:
        raise ValueError(err_msg)
    # END validate container object ** * ** * ** * ** * ** * ** * ** *

    # validate first position ** * ** * ** * ** * ** * ** * ** * ** * **
    # (i) a list of str values

    try:
        iter(_string_param_value[0])
    except:
        raise TypeError(err_msg)

    if isinstance(_string_param_value[0], (set, dict, str)):
        raise TypeError(err_msg)

    _string_param_value[0] = list(_string_param_value[0])

    if len(_string_param_value[0]) == 0:
        raise ValueError(err_msg)

    for item in _string_param_value[0]:
        if not isinstance(item, (str, type(None))):
            raise TypeError(err_msg)

    # END validate first position ** * ** * ** * ** * ** * ** * ** * **

    # validate second position ** * ** * ** * ** * ** * ** * ** * ** * *

    if _string_param_value[1] is None:
        # A LARGE NUMBER OF PASSES THAT WILL NEVER BE REACHED
        _string_param_value[1] = 1_000_000

    if 'INT' not in str(type(_string_param_value[1])).upper():
        raise TypeError(err_msg)

    if _string_param_value[1] < 1:
        raise ValueError(err_msg)

    # END validate second position ** * ** * ** * ** * ** * ** * ** * **

    # validate third position ** * ** * ** * ** * ** * ** * ** * ** * **

    if not isinstance(_string_param_value[2], str):
        raise TypeError(err_msg)

    _string_param_value[2] = _string_param_value[2].lower()

    if _string_param_value[2] != 'string':
        raise ValueError(err_msg)

    # END validate third position ** * ** * ** * ** * ** * ** * ** * **

    del err_msg

    return _string_param_value












