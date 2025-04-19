# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
from copy import deepcopy

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _get_next_param_grid._string._string import _string



@pytest.fixture
def _param_value():
    return [['saga', 'lbfgs'], 2, 'string']

@pytest.fixture
def _grids():
    return {0: {'a': ['saga', 'lbfgs'], 'b': [1,2,3]}, 1: {}}

@pytest.fixture
def _best_params():
    return {'a': 'saga', 'b': 2}


class TestString:


    def test_accuracy_1(self, _param_value, _grids, _best_params):
        # _pass (zero-indexed) is less than shrink pass - 1 (1-indexed)

        out = _string(_param_value, _grids[0]['a'], 0, _best_params['a'])

        assert out == ['saga', 'lbfgs']


    @pytest.mark.parametrize('_shrink_pass', (1, 2))
    def test_accuracy_2(self, _param_value, _grids, _best_params, _shrink_pass):
        # _pass (zero-indexed) is gte shrink pass - 1 (1 indexed)

        _grids[1] = deepcopy(_grids[0])
        _grids[2] = {}

        out = _string(_param_value, _grids[0]['a'], _shrink_pass, _best_params['a'])

        assert out == ['saga']



















