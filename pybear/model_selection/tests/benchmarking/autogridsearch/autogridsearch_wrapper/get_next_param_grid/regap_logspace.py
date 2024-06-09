# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _regap_logspace import _regap_logspace




_GRIDS = {
    0: {'a': ['a', 'b', 'c'], 'b': [1e2, 1e4, 1e6]},
    1: {'a': ['a', 'b', 'c'], 'b': [1e2, 1e4, 1e6]}
}

_IS_LOGSPACE = {
    'a': False,
    'b': 2.0
}

_params = {
    'a': [['a', 'b', 'c'], [3, 3, 3], 'string'],
    'b': [[1e2, 1e4, 1e6], [3, 3, 3], 'soft_float']
}

_pass = 1

_best_params_from_previous_pass = {
    'a': 'b',
    'b': 1e4
}

out_grids, out_params, out_is_logspace = \
    _regap_logspace(
        _GRIDS,
        _IS_LOGSPACE,
        _params,
        _pass,
        _best_params_from_previous_pass
    )

print(out_grids)
print(out_params)
print(out_is_logspace)















