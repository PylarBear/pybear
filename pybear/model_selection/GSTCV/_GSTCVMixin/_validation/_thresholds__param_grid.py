# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Iterable, Union

import numpy as np

from model_selection.GSTCV._type_aliases import (
    ParamGridType
)

from ._threshold_checker import _threshold_checker




def _validate_thresholds__param_grid(
    _thresholds: Union[Iterable[Union[int, float]], Union[int, float], None],
    _param_grid: Union[ParamGridType, Iterable[ParamGridType], None]
    ) -> list[ParamGridType]:


    err_msg = (f"param_grid must be a (1 - dictionary) or (2 - a list of "
            f"dictionaries) with strings as keys and lists as values")

    if _param_grid is None:
        _out_param_grid = [{}]
    elif isinstance(_param_grid, dict):
        _out_param_grid = [_param_grid]
    elif isinstance(_param_grid, str):
        raise TypeError(err_msg)
    else:
        try:
            _out_param_grid = list(_param_grid)
        except:
            raise TypeError(err_msg)

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

        if sum(map(lambda x: 'threshold' in x, list(map(str.lower, _grid)))) > 1:
            raise ValueError(
                f"there are multiple keys in param_dict[{grid_idx}] "
                f"indicating threshold"
            )

        new_grid = {}
        for _key, _value in _grid.items():
            if 'threshold' in _key.lower():
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
























