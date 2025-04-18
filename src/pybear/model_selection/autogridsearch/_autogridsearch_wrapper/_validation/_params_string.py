# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union, TypeAlias


# see _type_aliases, subtypes for DataType, GridType, PointsType, ParamType
DataType: TypeAlias = Union[None, str]
InStrGridType: TypeAlias = Sequence[DataType]
InStrPointsType: TypeAlias = Union[None, int]
InStrParamType: TypeAlias = Sequence[InStrGridType, InStrPointsType, str]
OutStrGridType: TypeAlias = list[DataType]
OutStrPointsType: TypeAlias = int
OutStrParamType: TypeAlias = list[OutStrGridType, OutStrPointsType, str]


def _string_param_value(
    _string_param_key: str,
    _string_param_value: InStrParamType
) -> OutStrParamType:

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


    # validate container object ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = (f"string_param '{_string_param_key}' -- _params values must "
        f"be list-like")
    try:
        iter(_string_param_value)
    except:
        raise TypeError(err_msg)

    if isinstance(_string_param_value, (set, dict, str)):
        raise TypeError(err_msg)

    del err_msg

    _string_param_value = list(_string_param_value)

    if len(_string_param_value) != 3:
        raise ValueError(f"string_param '{_string_param_key}' -- _params values "
            f"\n must contain 3 things --- first grid, shrink pass, the string "
            f"'string'")
    # END validate container object ** * ** * ** * ** * ** * ** * ** *

    # validate first position ** * ** * ** * ** * ** * ** * ** * ** * **
    # (i) a list of str values

    err_msg = (f"string_param '{_string_param_key}' -- first grid must be a "
        f"\nlist-like holding the grid-search values; cannot be empty and must "
        f"\ncontain the values (either strings or None-type) for its respective "
        f"\narg/kwarg of the estimator")

    try:
        iter(_string_param_value[0])
    except:
        raise TypeError(err_msg)

    if isinstance(_string_param_value[0], (dict, str)):
        raise TypeError(err_msg)

    _string_param_value[0] = list(_string_param_value[0])

    if len(_string_param_value[0]) == 0:
        raise ValueError(err_msg)

    for item in _string_param_value[0]:
        if not isinstance(item, (str, type(None))):
            raise TypeError(err_msg)

    del err_msg

    # END validate first position ** * ** * ** * ** * ** * ** * ** * **

    # validate second position ** * ** * ** * ** * ** * ** * ** * ** * *

    err_msg = (f"string_param '{_string_param_key}' -- 'shrink pass' must be "
        f"\nNone or an integer > 1 indicating the pass on which to reduce a "
        f"\nparam's grid to only a single value")

    if _string_param_value[1] is None:
        # A LARGE NUMBER OF PASSES THAT WILL NEVER BE REACHED
        _string_param_value[1] = 1_000_000

    try:
        float(_string_param_value[1])
        if isinstance(_string_param_value[1], bool):
            raise Exception
    except:
        raise TypeError(err_msg)

    if int(_string_param_value[1]) != _string_param_value[1]:
        raise TypeError(err_msg)

    _string_param_value[1] = int(_string_param_value[1])

    if _string_param_value[1] < 2:
        raise ValueError(err_msg)

    del err_msg

    # END validate second position ** * ** * ** * ** * ** * ** * ** * **

    # validate third position ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = (f"string_param '{_string_param_key}' -- final position in _params "
           f"\nmust be a string-like that says the word 'string'")

    if not isinstance(_string_param_value[2], str):
        raise TypeError(err_msg)

    _string_param_value[2] = _string_param_value[2].lower()

    if _string_param_value[2] != 'string':
        raise ValueError(err_msg)

    # END validate third position ** * ** * ** * ** * ** * ** * ** * **

    del err_msg

    return _string_param_value












