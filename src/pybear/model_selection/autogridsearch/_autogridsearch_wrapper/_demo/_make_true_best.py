# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from .._type_aliases import ParamsType, BestParamsType


def _make_true_best(
        _params: ParamsType
    ) -> BestParamsType:

    """
    Build a mock best_params_ with realistic values based on the grid-
    building instructions in _params.

    Parameters
    ----------
    _params: dict[str, list[...]] - grid-building instruction for all
    parameters

    Return
    ------
    -
        _true_best_params: dict[str, [int, float, str]] - mock best
        GridSearchCV results in format identical to sklearn / dask
        GridSearchCV.best_params_

    """

    _true_best_params = dict()

    for _param in _params:

        _single_param = _params[_param]
        _param_grid = _single_param[0]
        _type = _single_param[-1].lower()
        _min = None
        _max = None
        _gap = None

        if 'bool' in _type or 'string' in _type:
            _best = _param_grid[np.random.randint(0,len(_param_grid))]
        else:
            _min = min(_param_grid)
            _max = max(_param_grid)
            _gap = _max - _min
            if _type == 'hard_float':
                _best = int(np.random.uniform(_min, _max, size=(1,))[0])
            elif _type == 'hard_integer':
                _allowable = np.arange(_min, _max + 1, 1, dtype=np.int32)
                _best = int(np.random.choice(_allowable, 1, replace=False)[0])
                del _allowable
            elif _type in ['fixed_float', 'fixed_integer']:
                _best = int(np.random.choice(_param_grid, 1, replace=False)[0])
            elif _type == 'soft_float':
                _new_min = max(_min - _gap, 0)
                _new_max = _max + _gap
                _best = int(np.random.uniform(_new_min, _new_max, size=(1,))[0])
                del _new_min, _new_max
            elif _type == 'soft_integer':
                _new_min = max(_min - _gap, 1)
                _new_max = _max + _gap
                _new_points = int(_new_max - _new_min) + 1
                _new_grid = np.arange(_new_min, _new_max, _new_points).astype(int)[0],
                del _new_min, _new_max, _new_points
                _best = int(np.random.choice(_new_grid, 1, replace=False)[0])

        _true_best_params[_param] = _best


    del _single_param, _param_grid, _type, _min, _max, _gap

    return _true_best_params


























