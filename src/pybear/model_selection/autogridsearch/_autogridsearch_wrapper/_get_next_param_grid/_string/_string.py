# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable
from typing_extensions import TypeAlias


# see _type_aliases

# subtypes for str only
StrDataType: TypeAlias = str  # DataType sub
StrGridType: TypeAlias = Iterable[StrDataType] # GridType sub
StrPointsType: TypeAlias = int # PointsType sub
StrParamType: TypeAlias = list[StrGridType, StrPointsType, str] # ParamType sub




def _string(
            _param_value: StrParamType,
            _grid: StrGridType,
            _pass: int,
            _best_param_from_previous_pass: StrDataType
    ) -> list[StrDataType]:

    """
    Create the current round's search grid for a string parameter based
    on results from _best_params.

    Parameters
    ----------
    _param_value:
        list[Iterable[str], int, str] - string parameter grid instructions
    _grid:
        Iterable[str] - previous round's gridsearch values for string
        parameter
    _pass:
        int - zero-indexed count of passes to this point, inclusive; the
        current pass
    _best_param_from_previous_pass:
        str - best value returned from sklearn / dask best_params_

    Return
    ------
    -
        _grid: list[str] - new search grid for the current pass

    """


    # pass is zero-indexed, _param_value[1] is not
    if _pass >= _param_value[1] - 1:
        # _best_param_from_previous_pass] is a single value, wrap with []
        _grid = [_best_param_from_previous_pass]
    else:
        _grid = _param_value[0]


    return _grid
