# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _int._int_linspace_gap_gt_1 import _int_linspace_gap_gt_1




_SINGLE_GRID = [1,6,10]
_posn = 1
_is_hard = False
_hard_min = 1
_hard_max = 20
_points = 4

out_grid = _int_linspace_gap_gt_1(
    _SINGLE_GRID,
    _posn,
    _is_hard,
    _hard_min,
    _hard_max,
    _points
)

print(out_grid)









