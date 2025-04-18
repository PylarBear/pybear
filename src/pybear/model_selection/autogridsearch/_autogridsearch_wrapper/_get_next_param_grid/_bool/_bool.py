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
) -> list[BoolDataType]:

    """
    Create the current round's search grid for a boolean parameter based
    on results from _best_params.


    Parameters
    ----------
    _param_value:
        list[Sequence[bool], int, str] - boolean parameter grid
        instructions
    _grid:
        Sequence[bool] - previous round's gridsearch values for boolean
        parameter
    _pass:
        numbers.Integral - zero-indexed count of passes to this point,
        inclusive; the current pass
    _best_param_from_previous_pass:
        bool - best value returned from sklearn / dask best_params_


    Return
    ------
    -
        _grid: list[bool] - new search grid for the current pass


    """


    # pass is zero-indexed, _param_value[1] is not
    if _pass >= _param_value[1] - 1:
        # _best_param_from_previous_pass] is a single value, wrap with []
        _grid = [_best_param_from_previous_pass]
    else:
        _grid = _param_value[0]


    return _grid
