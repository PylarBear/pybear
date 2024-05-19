# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _string._string import _string



@pytest.fixture
def _instructions():
    return [['saga', 'lbfgs'], 2, 'string']

@pytest.fixture
def _grids():
    return {0: {'a': ['saga', 'lbfgs'], 'b': [1,2,3]}, 1: {}}

@pytest.fixture
def _best_params():
    return {'a': 'saga', 'b': 2}


class TestString:

    def test_rejects_a_param_not_already_in_GRIDS(self, _instructions,
                                                  _grids, _best_params):

        with pytest.raises(ValueError):
            _string('c', _instructions, _grids, 1, _best_params)


    def test_rejects_key_not_in_GRIDS(self, _instructions, _grids, _best_params):

        _grids[1] = _grids[0]

        with pytest.raises(ValueError):
            _string('a', _instructions, _grids, 2, _best_params)


    def test_accuracy_1(self, _instructions, _grids, _best_params):
        # _pass is less than transition point

        assert _string('a', _instructions, _grids, 1, _best_params) == \
               {0: {'a': ['saga', 'lbfgs'], 'b': [1,2,3]},
                1: {'a': ['saga', 'lbfgs']}} # <-- not getting num update here

    def test_accuracy_2(self, _instructions, _grids, _best_params):
        # _pass is gte transition point

        _grids[1] = {'a': ['saga', 'lbfgs'], 'b': [1,2,3]}
        _grids[2] = {}

        assert _string('a', _instructions, _grids, 2, _best_params) == \
               {0: {'a': ['saga', 'lbfgs'], 'b': [1, 2, 3]},
                1: {'a': ['saga', 'lbfgs'], 'b': [1, 2, 3]},
                2: {'a': ['saga']}}   # <-- not getting num update here







