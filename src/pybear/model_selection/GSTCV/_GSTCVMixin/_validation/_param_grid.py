# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import (
    ParamGridType,
    ParamGridsType
)

from ._param_grid_helper import _val_param_grid_helper

# pizza the type hints in here need work

def _val_param_grid(
    _param_grid: Union[ParamGridType, ParamGridsType, None]  # pizza y is this allowed to be None
) -> None:

    """
    Validate `param_grid` and any `thresholds` that may have been passed
    inside. `param_grid` can be a single param_grid or a list-like of
    param_grids.

    Validate format(s) is/are dict[str, list-like]. Validate `thresholds`
    is a list-like of numbers, with numbers in [0, 1] interval.


    Parameters
    ----------
    _param_grid:
        Union[ParamGridType, ParamGridsType, None] - A
        param_grid is a dictionary with hyperparameter names (str) as
        keys and list-likes of hyperparameter settings to test as values.
        `_param_grid` can be None, one of the described param_grids, or
        a list-like of such param_grids.


    Return
    ------
    -
        None


    """


    # pizza finalize the None issue
    _err_msg = (
        f"param_grid must be None, a (1 - dictionary) or (2 - a list-like "
        f"of dictionaries). \nthe dictionary keys must be strings and the "
        f"dictionary values must be list-like."
    )

    if _param_grid is None:
        return

    try:
        iter(_param_grid)
        if isinstance(_param_grid, str):
            raise Exception
    except Exception as e:
        raise TypeError(_err_msg)

    # _param_grid must be iter
    if isinstance(_param_grid, dict):
        if len(_param_grid) == 0:
            return
        _dpg = [_param_grid]   # _dpg = dum_param_grid
    else:
        _dpg = list(_param_grid)
    # _dpg must be list[non-empty dict] or list[some non-string iterables]


    for _grid_idx, _grid in enumerate(_dpg):

        _val_param_grid_helper(_grid, _grid_idx)






