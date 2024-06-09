# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from model_selection.autogridsearch._autogridsearch_wrapper._demo. \
    _display_true_best import _display_true_best



_demo_cls_params = {
    'a': [[1, 2, 3], [3, 3 ,3], 'fixed_integer'],
    'b': [['a', 'b', 'c'], 3, 'string'],
    'c': [[0, 50, 100], [3, 11, 11], 'soft_float'],
    'd': [['egg', 'ham', 'toast'], 3, 'string']
}

_true_best = {
    'a': 1,
    'b': 'c',
    'c': 50,
    'd': 'ham'
}


_display_true_best(_demo_cls_params, _true_best)











