# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _int._int_linspace_unit_gap import _int_linspace_unit_gap


# _posn = 'right', _is_hard = True, _hard_min = 1, _hard_max = 15, _points = 3

_SINGLE_GRID = [2,3,4]
_posn = 2
_is_hard = False
_hard_min = 1
_hard_max = 4
_points = 4

out_grid = _int_linspace_unit_gap(
                                    _SINGLE_GRID,
                                    _posn,
                                    _is_hard,
                                    _hard_min,
                                    _hard_max,
                                    _points
        )

print(out_grid)








