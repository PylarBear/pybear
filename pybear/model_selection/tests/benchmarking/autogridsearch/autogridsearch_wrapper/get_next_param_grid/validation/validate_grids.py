# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _validation._validate_grids import _validate_grids


a = {
    0: {'alpha': [1.0, 100.0, 10000.0], 'max_iter': [1, 100, 10000]},
    1: {'alpha': [100.0], 'max_iter': [100]}
}

_validate_grids(a)


assert map(isinstance, a.values(), ((int, float, str, bool) for _ in a))




















