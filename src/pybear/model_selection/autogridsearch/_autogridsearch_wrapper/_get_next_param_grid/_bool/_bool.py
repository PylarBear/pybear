# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases_bool import (
    BoolDataType,
    BoolParamType,
    BoolGridType
)



def _bool(
    _param_value: BoolParamType,
    _grid: BoolGridType,
    _pass: int,
    _best_param_from_previous_pass: BoolDataType
) -> BoolGridType:

    """
    Create the current round's search grid for a boolean parameter based
    on results from _best_params.


    Parameters
    ----------
    _param_value:
        BoolParamType - boolean parameter grid instructions
    _grid:
        BoolGridType - previous round's gridsearch values for boolean
        parameter
    _pass:
        int - zero-indexed count of passes to this point, inclusive; the
        current pass
    _best_param_from_previous_pass:
        BoolDataType - best value returned from GSCV best_params_


    Return
    ------
    -
        _grid: BoolGridType - new search grid for the current pass


    """


    # pass is zero-indexed
    if _param_value[1][_pass] == 1:
        # _best_param_from_previous_pass] is a single value, wrap with []
        _grid = [_best_param_from_previous_pass]
    else:
        _grid = _param_value[0]


    return _grid




