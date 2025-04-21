# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_bool import (
    InBoolParamType,
    BoolParamType
)

import numbers



def _cond_bool_param_value(
    _bool_param_value: InBoolParamType,
    _inf_shrink_pass: numbers.Integral
) -> BoolParamType:

    """
    Standardize format.

    COMES IN AS
    list-like(
        list-like('grid_value1', etc.),
        None or integer > 0,
        Literal['bool']
    )

    GOES OUT AS
    [
        list['grid_value1', etc.],
        1_000_000 or integer > 0,
        Literal['bool']
    ]


    Parameters
    ----------
    _bool_param_value:
        InBoolParamType - the 'params' dict value for a boolean parameter
        to be conditioned and standardized.
    _inf_shrink_pass:
        numbers.Integral - the value to be assigned for 'shrink pass' if
        the user entered None for that value.


    Returns
    -------
    -
        BoolParamType: the conditioned boolean parameter 'params' dict
        value.

    """


    if _bool_param_value[1] is None:
        # A LARGE NUMBER OF PASSES THAT WILL NEVER BE REACHED
        # this cannot be set to float('inf') because validated to be int
        assert isinstance(_inf_shrink_pass, numbers.Integral)
        assert _inf_shrink_pass >= 1
        _bool_param_value[1] = _inf_shrink_pass

    _bool_param_value[1] = int(_bool_param_value[1])

    # -- -- -- -- -- -- -- -- -- -- -- -- -- --


    return _bool_param_value




