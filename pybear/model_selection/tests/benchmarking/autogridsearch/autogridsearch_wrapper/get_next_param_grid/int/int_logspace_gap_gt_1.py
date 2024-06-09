# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _int._int_logspace_gap_gt_1 import _int_logspace_gap_gt_1




_SINGLE_GRID = [1e1,1e3,1e5]
_posn = 2
_is_logspace = 2.0
_is_hard = False
_hard_min = 1
_hard_max = 1e5
_points = 50

out_grid = _int_logspace_gap_gt_1(
    _SINGLE_GRID,
    _posn,
    _is_logspace,
    _is_hard,
    _hard_min,
    _hard_max,
    _points
)

print(out_grid)









