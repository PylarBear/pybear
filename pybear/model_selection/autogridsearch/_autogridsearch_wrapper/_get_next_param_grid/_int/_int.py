# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np


# RULES SURROUNDING LOGPACES AND INTEGERS
# --- CAN ONLY BE A "LOGSPACE" IF len(GRID) >= 3
# --- CAN ONLY BE A "LOGSPACE" IF log10 GAPS ARE UNIFORM (24_02_08_15_39_00 AND == 1)
# --- "FIXED" IN THIS ALGORITHM CANNOT BE LOGSPACE
# --- LOGSPACE IS POSITIVE DEFINITE IN LINEAR SPACE
# --- IF A SOFT LOGSPACE, REGAP CODE FORCES log10 GAPS TO <= 1
# --- INTEGER VALUES IN LINEAR SPACE IS ENFORCED BY VALIDATION IN __init__
# --- IF LINEAR GAP == 1, CANT DRILL ANY DEEPER


from typing import Union, TypeAlias
from ._int_linspace_unit_gap import _int_linspace_unit_gap
from ._int_linspace_gap_gt_1 import _int_linspace_gap_gt_1
from ._int_logspace_unit_gap import _int_logspace_unit_gap
from ._int_logspace_gap_gt_1 import _int_logspace_gap_gt_1


# see _type_aliases; subtypes for DataType, GridType
IntDataType: TypeAlias = int
IntGridType: TypeAlias = \
    Union[list[IntDataType], tuple[IntDataType], set[IntDataType]]


def _int(
            _SINGLE_GRID: IntGridType,
            _is_logspace: Union[bool, float],
            _posn: int,
            _is_hard: bool,
            _hard_min: IntDataType,
            _hard_max: IntDataType,
            _points: int
    ) -> tuple[list[IntDataType], Union[bool, float]]:


    """
    Take in a integer's grid from the last round of GridSearch along
    with the index position of the best value within that grid and
    return a new grid for the upcoming (current pass') GridSearch.
    Important factors in building the next grid: hard/soft,
    linspace/logspace, number of points.

    Parameters
    ----------
    _SINGLE_GRID:
        Union[list[int], tuple[int], set[int]] - The last round's search
        grid for a single param. May be linspace or logspace. _SINGLE_GRID
        must be sorted ascending, and is presumed to be by
        _validation._numerical_params (at least initially).
    _is_logspace:
        Union[bool, float] - For numerical params, if the space is linear,
        or some other non-standard interval, it is False. If it is
        logspace, the 'truth' of being a logspace is represented by a
        number indicating the interval of the logspace. E.g.,
        np.logspace(-5, 5, 11) would be represented by 1.0, and
        np.logspace(-20, 20, 9) would be represented by 5.0.
    _posn:
        int - the index position in the previous round's grid where
        the best value fell
    _is_hard:
        bool - whether the parameter has hard left and right boundaries.
        This field is read from the dtype/search field in _params. If
        hard, the left and right bounds are set from the lowest and
        highest values in the first round's search grid (the grid that
        is in _params.)
    _hard_min:
        int - if hard, the minimum value in the first round's search grid,
        otherwise ignored.
    _hard_max:
        int - if hard, the maximum value in the first round's search grid,
        otherwise ignored.
    _points:
        int - the number of points for this parameter's grid in the next
        pass of GridSearchCV, as read from _params.

    Return
    ------
    -
        _NEW_GRID:
        list[int] - new search grid for the current pass' upcoming
        search.

        _is_logspace:
        Union[bool, float] - current float parameter grid space is /
        is not logspace. All params leaving this module should be
        in linspace and the return value should always be False.


    """

    # cannot reach here if is 'fixed' or next pass has one point

    # 24_05_19_16_49_00 notes on _int_logspace_gap_gt_1 and
    # _int_logspace_unit_gap. Both point to the same core algorithm, but
    # are maintained here as separate modules with different validation
    # wrapped around the core. The original prediction was that unit and
    # >1 log gaps would need separate modules, and tests and validation
    # modules were built in anticipation of that. But now that both could
    # be passed to the same module, consolidating both operations for
    # unit and >1 gaps to that module would require appreciable overhaul
    # of existing tests and validation modules to make the tests and
    # validation handle the passing of both to the same place. So,
    # creating a 'core' module with functional code extracted from
    # _int_logspace_gap_gt_1 that feeds both unit and >1 log gap modules,
    # with individual validation for each of the modules.

    _LOGSPACE_PARAMS = (_SINGLE_GRID, _is_logspace, _posn, _is_hard,
                        _hard_min, _hard_max, _points)
    _LINSPACE_PARAMS = (_SINGLE_GRID, _posn, _is_hard, _hard_min,
                        _hard_max, _points)

    if not _is_logspace:

        if _posn == 0:
            _gap = _SINGLE_GRID[1] - _SINGLE_GRID[0]
        elif _posn == len(_SINGLE_GRID) - 1:
            _gap = _SINGLE_GRID[-1] - _SINGLE_GRID[-2]
        else:
            _gap = (_SINGLE_GRID[_posn + 1] - _SINGLE_GRID[_posn - 1]) / 2

        if _gap == 1:
            _OUT_GRID = _int_linspace_unit_gap(*_LINSPACE_PARAMS)
        else:
            _OUT_GRID = _int_linspace_gap_gt_1(*_LINSPACE_PARAMS)

        del _gap



    elif _is_logspace:

        # THIS CAN ONLY BE ACCESSED ON THE FIRST PASS AFTER SHIFTER

        _LOG_SINGLE_GRID = np.log10(_SINGLE_GRID)

        if _posn == 0:
            _log_gap = _LOG_SINGLE_GRID[1] - _LOG_SINGLE_GRID[0]
        elif _posn == len(_SINGLE_GRID) - 1:
            _log_gap = _LOG_SINGLE_GRID[-1] - _LOG_SINGLE_GRID[-2]
        else:
            _log_gap = (_LOG_SINGLE_GRID[_posn + 1] - _LOG_SINGLE_GRID[_posn - 1]) / 2


        if _log_gap == 1:
            _OUT_GRID = _int_logspace_unit_gap(*_LOGSPACE_PARAMS)
        else:
            _OUT_GRID = _int_logspace_gap_gt_1(*_LOGSPACE_PARAMS)

        del _log_gap

        _is_logspace = False


    del _LOGSPACE_PARAMS, _LINSPACE_PARAMS


    return _OUT_GRID, _is_logspace


















