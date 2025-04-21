# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_str import (
    InStrParamType,
    StrParamType
)

import numbers



def _cond_string_param_value(
    _string_param_value: InStrParamType,
    _inf_shrink_pass: numbers.Integral
) -> StrParamType:

    """
    Standardize format.

    COMES IN AS
    list-like(
        list-like('grid_value1', 'grid_value2', etc.),
        None or integer > 0,
        Literal['string']
    )

    GOES OUT AS
    [
        list['grid_value1', 'grid_value2', etc.],
        _inf_shrink_pass or integer > 0,
        Literal['string']
    ]


    Parameters
    ----------
    _string_param_value:
        InStrParamType - the 'params' dict value for a string parameter
        to be conditioned and standardized.
    _inf_shrink_pass:
        numbers.Integral - the value to be assigned for 'shrink pass' if
        the user entered None for that value.


    Returns
    -------
    -
        StrParamType: the conditioned string parameter 'params' dict
        value.

    """


    if _string_param_value[1] is None:
        # A LARGE NUMBER OF PASSES THAT WILL NEVER BE REACHED
        # this cannot be set to float('inf') because validated to be int
        assert isinstance(_inf_shrink_pass, numbers.Integral)
        assert _inf_shrink_pass >= 1
        _string_param_value[1] = _inf_shrink_pass

    _string_param_value[1] = int(_string_param_value[1])

    # -- -- -- -- -- -- -- -- -- -- -- -- -- --


    return _string_param_value




