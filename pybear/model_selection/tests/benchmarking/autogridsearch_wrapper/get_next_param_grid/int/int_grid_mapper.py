# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _int._int_grid_mapper import _int_grid_mapper


_left = 1
_right = 5
_points = 8
for (_left, _right, _points) in (
                                (1,5,8),
                                (1,2,2),
                                (1,3,3),
                                (1,4,4),
                                (10,50,7),
                                (7,37,13)
    ):

    out_grid = _int_grid_mapper(
                                _left,
                                _right,
                                _points
            )

    print()
    print(f"** * " * 10)
    print(f"_left = {_left}, _right = {_right}, _points = {_points}")
    print(out_grid)
    print(f"actual points = {len(out_grid)}")


