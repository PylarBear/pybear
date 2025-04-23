# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _print_results import _print_results



_params = {
    'a': [['x', 'y', 'z'], [3, 3 ,3, 3], 'string'],
    'b': [[1, 2, 3, 4], [4, 4, 4, 3], 'fixed_integer'],
    'c': [[25, 30, 35], [3, 3, 6, 3], 'soft_float']
}


_true_best = {
    'a': 'x',
    'b': 4,
    'c': 28.8205373
}


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
    },
    2: {
        'a': ['x', 'y', 'z'],
        'b': [1, 2, 3, 4],
        'c': [25, 27, 29, 31, 33, 35]
    },
    3: {
        'a': ['x', 'y', 'z'],
        'b': [1, 2, 3, 4],
        'c': [28, 29, 30]
    }
}


_RESULTS = {
    0: {
        'a': ['x'],
        'b': 4,
        'c': 30
    },
    1: {
        'a': ['x'],
        'b': 4,
        'c': 30
    },
    2: {
        'a': ['x'],
        'b': 4,
        'c': 29
    },
    3: {
        'a': ['x'],
        'b': 4,
        'c': 29
    }
}


_print_results(_GRIDS, _RESULTS)







