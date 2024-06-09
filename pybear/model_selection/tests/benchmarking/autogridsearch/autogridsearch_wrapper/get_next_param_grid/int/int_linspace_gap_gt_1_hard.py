# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _int._int_linspace_gap_gt_1_hard import _int_linspace_gap_gt_1_hard




_SINGLE_GRID = [2,3,4]
_posn = 2
_hard_min = -1
_hard_max = 20
_points = 50

_left, _right = _int_linspace_gap_gt_1_hard(
    _SINGLE_GRID,
    _posn,
    _hard_min,
    _hard_max,
)

print(_left, _right)









