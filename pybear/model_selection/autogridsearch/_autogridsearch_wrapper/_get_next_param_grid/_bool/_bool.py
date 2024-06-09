# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable, TypeAlias


# see _type_aliases

# subtypes for bool only
BoolDataType: TypeAlias = bool  # DataType sub
BoolGridType: TypeAlias = Iterable[BoolDataType]  # GridType sub
BoolPointsType: TypeAlias = int  # PointsType sub
BoolParamType: TypeAlias = list[BoolGridType, BoolPointsType, str] # ParamType sub




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
        list[Iterable[bool], int, str] - boolean parameter grid instructions
    _grid:
        Iterable[bool] - previous round's gridsearch values for boolean
        parameter
    _pass:
        int - zero-indexed count of passes to this point, inclusive; the
        current pass
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
