# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np


from _validate_best_params import _validate_best_params
from _update_phlite import _update_phlite
from _shift._shift import _shift
from _regap_logspace import _regap_logspace
from _float._float import _float
from _int._int import _int
from _string._string import _string

def _get_next_param_grid(
                         _GRIDS: dict,
                         _params: dict,
                         _PHLITE: dict,
                         _IS_LOGSPACE: dict,
                         _best_params_from_previous_pass: dict,
                         _pass: int,
                         _total_passes: int,
                         _total_passes_is_hard: bool,
                         _shift_ctr: int,
                         _max_shifts: int
    ) -> dict:

    """
    # USE best_params_from_previous_pass AND THE LAST GRID USED TO BUILD
    A NEW GRID ####################

    Core functional method. This should not be reached on the first pass
    (pass zero). For subsequent passes, generate new grids based on the
    previous grid (as held in GRIDS) and its associated best_params_
    returned from GridSearchCV.

    Parameters
    ----------
    _pass:
        int - iteration counter
    :param
        _best_params_from_previous_pass: dict - best_params_ returned
        by Gridsearch for the previous pass

    Return
    ------
    pizza_grid:
        dict of grids to be passed to GridSearchCV for the next pass,
        must be in param_grid format

    """

    if len(_GRIDS) == 0 or len(_GRIDS[max(_GRIDS.keys())]) == 0:
        raise ValueError(f"an empty GRIDS has been passed to get_next_param_grid()")

    _validate_best_params( _GRIDS, _pass, _best_params_from_previous_pass)

    _GRIDS[_pass] = dict()


    # self.PARAM_HAS_LANDED_INSIDE_THE_EDGES is first defined in
    # autogridsearch_wrapper.reset()

    # must establish if a soft num param has fallen inside the edges of
    # its grid. string_parameter AND hard/fixed numerical_parameters
    # CANNOT BE ON AN EDGE (ALGORITHMICALLY SPEAKING)!
    # this is not needed after the pass where all soft num fall inside
    # the edges (all values in PHLITE will be False and cannot gain
    # re-entry to the place where they could be set back to True.)
    # Update PHLITE with the results from the last pass to assess the
    # current need for shifting.
    if False in _PHLITE.values() and _shift_ctr <= _max_shifts:

        if _shift_ctr < _max_shifts:

            _PHLITE = _update_phlite(
                                        _PHLITE,
                                        _GRIDS[_pass-1],
                                        _params,
                                        _best_params_from_previous_pass
            )

        elif _shift_ctr == _max_shifts:
            _PHLITE = {k: True for k in _PHLITE}

    # END UPDATE PHLITE ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # After update to PHLITE, if any params are still landing on the edges,
    # must slide their grids and rerun all the other params with their
    # same grids.
    if False in _PHLITE.values() and _shift_ctr < _max_shifts:

        _shift_ctr += 1

        if not _total_passes_is_hard:
            _total_passes += 1

        _GRIDS, _params = _shift(
                                    _GRIDS,
                                    _PHLITE,
                                    _IS_LOGSPACE,
                                    _params,
                                    _pass,
                                    _best_params_from_previous_pass,
                                    _total_passes_is_hard
        )

    # END SHIFT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # REGAP ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # 24_05_11_15_27_00 does _total_passes_is_hard matter here?

    if any([(v > 1 and _PHLITE.get(k, True)) for k, v in _IS_LOGSPACE.items()]) or \
            _shift_ctr == _max_shifts:

        # PIZZA 24_05_14_08_38_00, THIS IS CURRENTLY TAKING IN ALL PARAMS AND
        # REGAPPING EVERYTHING UNCONDITIONALLY. THIS WILL HAVE TO CHANGE TO
        # ONE AT A TIME CONDITIONALY ON IF _PHLITE[_param] == True

        # SOMETHING LIKE:
        # for _param in _GRIDS[_pass]:
        #     if _PHLITE.get(_param, True) is True:
        #         _GRIDS[_pass][_param], _params[_param], _IS_LOGSPACE[_param] = \
        #             _regap_logspace(PIZZA, PIZZA)


        _GRIDS, _params, _IS_LOGSPACE = \
            _regap_logspace(
                            _GRIDS,
                            _IS_LOGSPACE,
                            _params,
                            _pass,
                            _best_params_from_previous_pass
        )

        return _GRIDS, _params

    # END REGAP ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if False in _PHLITE.values() and _shift_ctr < _max_shifts:
        return _GRIDS, _params, _shift_ctr


    # IF REACHED THIS POINT:
    # (i) EVERYTHING IN PARAM_HAS_LANDED_INSIDE_THE_EDGES IS True,
    # (ii) ANY LOGSPACES HAVE AN INTERVAL OF <= 1 AND WILL CONVERT TO LINSPACES


    if any([v > 1 for k,v in _IS_LOGSPACE.items()]):
        ValueError(f"{_single_param}: an integer logspace with log10 gap > "
                   f"1 has made it into digging section")



    for hprmtr in _params:


        _best = _best_params_from_previous_pass[hprmtr]

        if _params[hprmtr][-1] == 'string':
            # 24_05_12_18_32_00 PIZZA REVISIT _string ABOUT ARGS
            # VERIFY THE ORDER IN THIS FOR LOOP IS CORRECT
            _GRIDS = _string(
                            [1e0, 1e5, 1e10, 1e15, 1e20],
                            hprmtr,
                            _params[hprmtr],
                            _GRIDS,
                            _pass,
                            _best_params_from_previous_pass
            )

            continue

        _type = _params[hprmtr][-1]
        _points = _params[hprmtr][1][_pass]
        _grid = _GRIDS[_pass - 1][hprmtr]

        _best_param_posn = np.isclose(_grid, _best, rtol=1e-6)

        if _best_param_posn.sum() != 1:
            ValueError(f"uniquely locating best_param position in search grid is "
               f"failing, should locate to 1 position, but locating to "
               f"{_best_param_posn.sum()} positions")

        _best_param_posn = np.arange(len(_grid))[_best_param_posn][0]

        # PIZZA from _int_logspace 24_05_13_09_23_00
        elif _posn not in range(len(_grid)):
            raise Exception(
                f"{hprmtr}: _posn ({_posn}) is not in range of _grid")

        if _points == 1:
            _GRIDS[_pass][hprmtr] = [_best]
            _IS_LOGSPACE[hprmtr] = False  # MAY HAVE ALREADY BEEN FALSE
            continue

        elif 'fixed' in _params[hprmtr][-1]:
            # THIS MUST BE AFTER _points == 1
            _GRIDS[_pass][hprmtr] = _params[hprmtr][0]
            continue

        if 'HARD' in _type.upper():
            _is_hard = True
        elif 'SOFT' in _type.upper():
            _is_hard = False
        else:
            raise ValueError(f"{hprmtr}: bound_type must contain 'hard' or 'soft' ({_type})")

        # ONLY NEEDED FOR 'hard' NUMERICAL
        _hard_min = _GRIDS[0][hprmtr][0]
        _hard_max = _GRIDS[0][hprmtr][-1]

        if 'integer' in _type:

            # PIZZA FIX 24_05_13_08_41_00
            # FROM INT LINSPACE
            if _pass == self.shift_ctr and not _SINGLE_GRID[0] == 1:
                raise Exception(f"{hprmtr}: a soft integer is on a left edge "
                                f"after shifter and value != 1")
            if _pass == self.shift_ctr:
                raise Exception(f"{hprmtr}: a soft integer is on a right "
                                f"edge... should have shifted")
            # FROM INT LOGSPACE
            if not is_hard and _grid[0] != 1:
                raise Exception(
                    f"{hprmtr}: a soft logspace integer is on a left edge and value != 1")
            if not is_hard:
                raise Exception(
                    f"{hprmtr}: a soft logspace integer is on a right edge... should have shifted")
            # END PIZZA FIX 24_05_13_08_41_00

            _grid, _is_logspace = \
                _int(
                    _GRIDS[_pass-1][hprmtr],
                    _IS_LOGSPACE[hprmtr],
                    _best_param_posn,
                    _is_hard,
                    _hard_min,
                    _hard_max,
                    _points
                )

            _GRIDS[_pass][hprmtr] = _grid
            _params[hprmtr][-2][_pass] = len(_grid)
            del _grid


        elif 'float' in _type:

            _grid, _is_logspace = _float(
                _GRIDS[_pass-1][hprmtr],
                _IS_LOGSPACE[hprmtr],
                _best_param_posn,
                _is_hard,
                _hard_min,
                _hard_max,
                _points
            )

            _GRIDS[_pass][hprmtr] = _grid
            _IS_LOGSPACE[hprmtr] = _is_logspace

            del _grid, _is_logspace

        # IF ANY ADJUSTMENTS WERE MADE TO _points, CAPTURE IN numerical_params
        _params[hprmtr][1][_pass] = _points

    try:
        del _best, _grid, _points, _type, is_hard, best_param_posn, hard_min, hard_max
    except:
        pass




    return _GRIDS















