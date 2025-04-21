# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection.autogridsearch._autogridsearch_wrapper._demo.\
    _mock_gscv import _mock_gscv



# prove out the part that displays info about the parameters


_GRIDS = {
    0: {
        'a': ['x', 'y', 'z'],
        'b': [1, 2, 3, 4],
        'c': [20, 30, 40]
    },
    1: {
        'a': ['x', 'y', 'z'],
        'b': [1, 2, 3, 4],
        'c': [25, 30, 35]
    }
}

_params = {
    'a': [['x', 'y', 'z'], [3, 3 ,3], 'string'],
    'b': [[1, 2, 3, 4], [4, 4, 4], 'fixed_integer'],
    'c': [[25, 30, 35], [3, 3, 6], 'soft_float']
}

_true_best = {
    'a': 'x',
    'b': 4,
    'c': 28.8205373
}

_best_params_round_zero = {}

_best_params_round_one = {
    'a': 'x',
    'b': 4,
    'c': 30
}


_pass = 0

_best_params_ = _mock_gscv(
    _GRIDS,
    _params,
    _true_best,
    _best_params_round_zero,
    _pass,
    _pause_time=1
)

print(f"best_params round 0:")
print(_best_params_)
print()


_pass = 1

_best_params_ = _mock_gscv(
    _GRIDS,
    _params,
    _true_best,
    _best_params_round_one,
    _pass,
    _pause_time=1
)

print(f"best_params round 1:")
print(_best_params_)
print()



