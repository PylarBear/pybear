# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from copy import deepcopy
import numpy as np


# PIZZA 24_05_14_08_42_00 THIS IS MOST LIKELY GOING TO NEED TO CHANGE TO
# HANDLING ONLY ONE PARAM AT A TIME

# ALSO PIZZA ABOUT hard_min AND hard_max?




def _regap_logspace(
                    _GRIDS: dict,
                    _IS_LOGSPACE: dict,
                    _params: dict,
                    _pass: int,
                    _best_params_from_previous_pass: dict
    ) -> [dict, dict, dict]:

    """

    If a logspace numerical parameter has log gap > 1 and has landed
    inside the edges of its grid (or is forced into here due to
    max_shifts), re-gap the logspace to 1.


    Parameters
    ----------
    _GRIDS:
        dict - holds i) all the param_grids run so far, and ii) a full
         grid for the current pass which will be modified here for the
         regapped params.
    _IS_LOGSPACE:
        dict - if a param is logspace and if so, what the log interval is
    _params:
        dict - search instructions and dtypes for each param
    _pass:
        int - the index of the upcoming gridsearch
    _best_params_from_previous_pass:
        dict - best_params_ from dask or sklearn GridsearchCV

    Return
    ------
    -
        _GRIDS: dict - _GRIDS updated with logspace intervals reduced to 1

        _params: dict - _params updated with points reflective of new
            points for logspace interval == 1

        _IS_LOGSPACE: dict - _IS_LOGSPACE updated with the unitized gaps

    """

    __GRIDS = deepcopy(_GRIDS)
    __IS_LOGSPACE = deepcopy(_IS_LOGSPACE)
    __params = deepcopy(_params)

    for _param in _IS_LOGSPACE:

        if not _IS_LOGSPACE[_param] > 1:
            continue

        _LOG_OLD_GRID = np.log10(__GRIDS[_pass - 1][_param])

        _log_best = np.log10(_best_params_from_previous_pass[_param])

        _log_gap = np.unique(_LOG_OLD_GRID[1:] - _LOG_OLD_GRID[:-1])

        if len(np.unique(_log_gap)) != 1:
            raise ValueError(f"{_param}: a logspace with a non-uniform gap")

        _log_gap = abs(_log_gap[0])

        # USE THE VALUES THAT FALL TO THE LEFT AND RIGHT OF
        # THE BEST VALUE TO CREATE A NEW RANGE WITH INCREMENT 1.
        _new_left = int(np.floor(_log_best - _log_gap))
        _new_right = int(np.ceil(_log_best + _log_gap))

        _points = abs(_new_right - _new_left) + 1

        _NEW_GRID = np.linspace(_new_left, _new_right, _points)
        _NEW_GRID = 10 ** _NEW_GRID
        _NEW_GRID = _NEW_GRID.tolist()
        if 'integer' in __params[_param][-1]:
            _NEW_GRID = list(map(int, _NEW_GRID))
        elif 'float' in __params[_param]:
            _NEW_GRID = list(map(float, _NEW_GRID))

        del _LOG_OLD_GRID, _new_left, _new_right

        # update GRIDS for the current param & pass
        __GRIDS[_pass][_param] = _NEW_GRID

        # OVERWRITE PARAM'S COMING PASS'S _points WITH POINTS FOR GAP==1
        __params[_param][1][_pass] = _points
        del _points

        # OVERWRITE _IS_LOGSPACE WITH NEW GAP
        __IS_LOGSPACE[_param] = 1.0

    return __GRIDS, __params, __IS_LOGSPACE













