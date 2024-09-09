# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union, Iterable
from typing_extensions import TypeAlias

# see _type_aliases, subtypes for DataType, GridType, PointsType, ParamType
BoolDataType: TypeAlias = bool
InBoolGridType: TypeAlias = \
    Union[list[BoolDataType], tuple[BoolDataType], set[BoolDataType]]
InBoolPointsType: TypeAlias = Union[None, int]
InBoolParamType: TypeAlias = Union[list[InBoolGridType, InBoolPointsType, str],
                                tuple[InBoolGridType, InBoolPointsType, str]]
OutBoolGridType: TypeAlias = list[BoolDataType]
OutBoolPointsType: TypeAlias = int
OutBoolParamType: TypeAlias = list[OutBoolGridType, OutBoolPointsType, str]


def _bool_param_value(
        _bool_param_key: str,
        _bool_param_value: InBoolParamType
    ) -> OutBoolParamType:

    """
    Validate _bool_param_value --- standardize format

    COMES IN AS
    list-like(
                list-like('grid_value1', etc.),
                None or integer > 0,
                'bool'
    )

    validate bool_params' dict value is a list-like that contains
    (i) a list-like of bool values
    (ii) a positive integer or None
    (iii) 'bool' (a string-type of the word 'bool')

    GOES OUT AS
    [
    ['grid_value1', etc.],
    1_000_000 or integer > 0,
    'bool'
    ]


    """

    if not isinstance(_bool_param_key, str):
        raise TypeError(f"_bool_param_key must be a string")


    # validate container object ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = (f"bool_param '{_bool_param_key}' -- _params values must "
        f"be list-like")
    try:
        iter(_bool_param_value)
    except:
        raise TypeError(err_msg)

    if isinstance(_bool_param_value, (set, dict, str)):
        raise TypeError(err_msg)

    del err_msg

    _bool_param_value = list(_bool_param_value)

    if len(_bool_param_value) != 3:
        raise ValueError(f"bool_param '{_bool_param_key}' -- _params values "
            f"\n must contain 3 things --- first grid, shrink pass, the string "
            f"'bool'")
    # END validate container object ** * ** * ** * ** * ** * ** * ** *

    # validate first position ** * ** * ** * ** * ** * ** * ** * ** * **
    # (i) a list of bool values

    err_msg = (f"bool_param '{_bool_param_key}' -- first grid must be a "
        f"\nlist-like holding the grid-search values; cannot be empty and must "
        f"\ncontain the values (booleans) for its respective arg/kwarg of the "
        f"estimator")

    try:
        iter(_bool_param_value[0])
    except:
        raise TypeError(err_msg)

    if isinstance(_bool_param_value[0], (dict, str)):
        raise TypeError(err_msg)

    _bool_param_value[0] = list(_bool_param_value[0])

    if len(_bool_param_value[0]) == 0:
        raise ValueError(err_msg)

    for item in _bool_param_value[0]:
        if not isinstance(item, bool):
            raise TypeError(err_msg)

    del err_msg

    # END validate first position ** * ** * ** * ** * ** * ** * ** * **

    # validate second position ** * ** * ** * ** * ** * ** * ** * ** * *

    err_msg = (f"bool_param '{_bool_param_key}' -- 'shrink pass' must be "
        f"\nNone or an integer > 1 indicating the pass on which to reduce a "
        f"\nparam's grid to only a single value")

    if _bool_param_value[1] is None:
        # A LARGE NUMBER OF PASSES THAT WILL NEVER BE REACHED
        _bool_param_value[1] = 1_000_000

    try:
        float(_bool_param_value[1])
        if isinstance(_bool_param_value[1], bool):
            raise Exception
    except:
        raise TypeError(err_msg)

    if int(_bool_param_value[1]) != _bool_param_value[1]:
        raise TypeError(err_msg)

    _bool_param_value[1] = int(_bool_param_value[1])

    if _bool_param_value[1] < 2:
        raise ValueError(err_msg)

    del err_msg

    # END validate second position ** * ** * ** * ** * ** * ** * ** * **

    # validate third position ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = (f"bool_param '{_bool_param_key}' -- final position in _params "
           f"\nmust be a string-like that says the word 'bool'")

    if not isinstance(_bool_param_value[2], str):
        raise TypeError(err_msg)

    _bool_param_value[2] = _bool_param_value[2].lower()

    if _bool_param_value[2] != 'bool':
        raise ValueError(err_msg)

    # END validate third position ** * ** * ** * ** * ** * ** * ** * **

    del err_msg

    return _bool_param_value












