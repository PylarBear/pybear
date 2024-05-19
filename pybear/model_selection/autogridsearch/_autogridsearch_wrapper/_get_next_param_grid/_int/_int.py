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



from _int_linspace_unit_gap import _int_linspace_unit_gap
from _int_linspace_gap_gt_1 import _int_linspace_gap_gt_1
from _int_logspace_unit_gap import _int_logspace_unit_gap
from _int_logspace_gap_gt_1 import _int_logspace_gap_gt_1






def _int(
            _SINGLE_GRID: list,
            _is_logspace: [bool, float],
            _posn: int,
            _is_hard: bool,
            _hard_min: int,
            _hard_max: int,
            _points: int
    ) -> [dict, [bool, float]]:







    """
    Take in a integer's grid from the last round of GridSearch along
    with the index position of the best value within that grid and
    return a new grid for the upcoming (current pass') GridSearch.
    Important factors in building the next grid: hard/soft,
    linspace/logspace, number of points.

    Parameters
    ----------
    _SINGLE_GRID:
        list - The last round's search grid for a single param.
        _SINGLE_GRID must be sorted ascending, and is presumed to be by
        _validation._numerical_params (at least initially).
    _is_logspace:
        [bool, float] - For numerical params, if the space is linear, or some
        other non-standard interval, it is False. If it is logspace, the 'truth'
        of being a logspace is represented by a number indicating the interval
        of the logspace. E.g., np.logspace(-5, 5, 11) would be represented by
        1.0, and np.logspace(-20, 20, 9) would be represented by 5.0.
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
        int - if hard, the minimum value in the first round's search grid.
    _hard_max:
        int - if hard, the maximum value in the first round's search grid.
    _points:
        int - the number of points for the current grid as read from _params.

    Return
    ------
    -
         _NEW_GRID:
            list - new search grid for the current pass' upcoming search.

         _is_logspace:
            bool, float - current float parameter grid space is / is not
            logspace. All params leaving this module should be linspace
            and the return value should always be False.



    """

    # cannot reach here if is 'fixed' or next pass has one point

    if not _is_logspace:

        match _posn:
            case 0:
                # ON THE LEFT EDGE
                _gap = _SINGLE_GRID[1] - _SINGLE_GRID[0]
                if _gap == 1:
                    _OUT_GRID = _int_linspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_linspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _gap

            case _posn if _posn == len(_SINGLE_GRID) - 1:
                # ON THE RIGHT EDGE
                _gap = _SINGLE_GRID[-1] - _SINGLE_GRID[-2]
                if _gap == 1:
                    _OUT_GRID = _int_linspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_linspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _gap

            case other:
                # IN THE MIDDLE SOMEWHERE
                _gap = (_SINGLE_GRID[_posn + 1] - _SINGLE_GRID[_posn - 1]) / 2
                if _gap == 1:
                    _OUT_GRID = _int_linspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_linspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _gap



    elif _is_logspace:

        # THIS CAN ONLY BE ACCESSED ON THE FIRST PASS AFTER SHIFTER

        _LOG_SINGLE_GRID = np.log10(_SINGLE_GRID)

        match _posn:
            case 0:
                # ON THE LEFT EDGE
                _log_gap = _LOG_SINGLE_GRID[1] - _LOG_SINGLE_GRID[0]
                if _log_gap == 1:
                    _OUT_GRID = _int_logspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_logspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _log_gap

            case _posn if _posn == len(_SINGLE_GRID) - 1:
                # ON THE RIGHT EDGE
                _log_gap = _LOG_SINGLE_GRID[-1] - _LOG_SINGLE_GRID[-2]
                if _log_gap == 1:
                    _OUT_GRID = _int_logspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_logspace,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_logspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_logspace,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _log_gap

            case other:
                # IN THE MIDDLE SOMEWHERE
                _log_gap = (_LOG_SINGLE_GRID[_posn + 1] - _LOG_SINGLE_GRID[_posn - 1]) / 2
                if _log_gap == 1:
                    _OUT_GRID = _int_logspace_unit_gap(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )
                else:
                    _OUT_GRID = _int_logspace_gap_gt_1(
                        _SINGLE_GRID,
                        _posn,
                        _is_hard,
                        _hard_min,
                        _hard_max,
                        _points
                    )

                del _log_gap




        _is_logspace = False






    return _OUT_GRID, _is_logspace


















