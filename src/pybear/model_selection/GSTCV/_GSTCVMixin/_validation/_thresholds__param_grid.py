# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Iterable
from typing_extensions import Union

import numpy as np

from ....GSTCV._type_aliases import (
    ParamGridType
)

from ._threshold_checker import _threshold_checker




def _validate_thresholds__param_grid(
    _thresholds: Union[Iterable[Union[int, float]], Union[int, float], None],
    _param_grid: Union[ParamGridType, Iterable[ParamGridType], None]
    ) -> list[ParamGridType]:

    """

    Jointly validate thresholds and param_grid, because once thresholds
    is validated, it is put inside the param grids during internal
    processing by GSTCV.

    thresholds - The decision threshold strategy to use when performing
    hyperparameter search.

    param_grid - Dictionary with parameters names (str) as keys and
    lists of parameter settings to try as values.

    Validate param grids. If param grids are passed, verify format is
    dict[str, list-like]. Get param_grid into list(dict[str, list-like])
    format. Validate thresholds is None (default), a number, or a list-
    like of numbers, with numbers in [0, 1] interval. Convert thresholds
    to a list-like of floats. If thresholds is None, use default
    thresholds. Put thresholds inside param grid(s).


    Parameters
    ----------

    _thresholds:
        Union[Iterable[Union[int, float]], Union[int, float], None] -

    _param_grid:
        Union[ParamGridType, Iterable[ParamGridType], None] -


    Return
    ------
        returns param grid (inside a list) with thresholds inside it,
        no matter how (or if) thresholds was passed.


    """



    err_msg = (f"param_grid must be a (1 - dictionary) or (2 - a list-like of "
        f"dictionaries). the dictionary keys must be strings and the "
        f"dictionary values must be vector-like")

    if _param_grid is None:
        _out_param_grid = [{}]
    else:
        try:
            iter(_param_grid)
            if isinstance(_param_grid, str):
                raise
        except:
            raise TypeError(err_msg)

        if len(_param_grid) == 0:
            _out_param_grid = [{}]
        elif isinstance(_param_grid, dict):
            _out_param_grid = [_param_grid]
        else:
            _out_param_grid = list(_param_grid)


    # param_grid must be list at this point
    for grid_idx, _grid in enumerate(_out_param_grid):
        if not isinstance(_grid, dict):
            raise TypeError(err_msg)
        for k, v in _grid.items():
            if not isinstance(k, str):
                raise TypeError(err_msg)

            try:
                iter(v)
                if isinstance(v, (str, dict)):
                    raise Exception
                v = list(v)
            except:
                raise TypeError(err_msg)

            _out_param_grid[grid_idx][k] = np.array(v)

    del err_msg

    # at this point param_grid must be a list of dictionaries having
    # str as keys and np arrays as values
    for grid_idx, _grid in enumerate(_out_param_grid):

        new_grid = {}
        for _key, _value in _grid.items():
            if _key.lower() == 'thresholds':
                new_grid['thresholds'] = _value
            else:
                new_grid[_key] = _value

        _grid = new_grid
        del new_grid

        if 'thresholds' in _grid:
            _grid['thresholds'] = \
                _threshold_checker(_grid['thresholds'], False, grid_idx)

        elif 'thresholds' not in _grid:
            _grid['thresholds'] = _threshold_checker(_thresholds, True, 0)

        _out_param_grid[grid_idx] = _grid

    del _grid


    return _out_param_grid
























