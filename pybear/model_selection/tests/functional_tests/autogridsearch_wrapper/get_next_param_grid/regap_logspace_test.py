# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _regap_logspace import _regap_logspace



@pytest.fixture
def good_params():
    return {
                'a': [['a','b','c','d'], 3, 'string'],
                'b': [[1,2,3,4], [4,4,4], 'fixed_integer'],
                'c': [[10,20,30,40], [4,4,4], 'fixed_float'],
                'd': [[1,100,10000], [3,3,3], 'soft_integer'],
                'e': [[25,50,75], [3,3,3], 'soft_integer'],
                'f': [[0, 0.5, 1.0], [3,3,3], 'hard_float'],
                'g': [[0,2,4,6], [4,4,4], 'hard_integer'],
                'h': [[1,10,100,1000], [4,4,4], 'soft_float'],
                'i': [[20,40,60,80], [4,4,4], 'soft_float'],
                'j': [[1e0, 1e4, 1e8, 1e12, 1e16], [5,5,5], 'soft_float']
    }


@pytest.fixture
def good_is_logspace():
    return {
            'a': False,
            'b': False,
            'c': False,
            'd': 2,
            'e': False,
            'f': False,
            'g': False,
            'h': 1,
            'i': False,
            'j': 4
    }


@pytest.fixture
def good_grids():
    return {
            0: {
                'a': ['a','b','c','d'],
                'b': [1,2,3,4],
                'c': [10,20,30,40],
                'd': [1,100,10000],
                'e': [25,50,75],
                'f': [0, 0.5, 1.0],
                'g': [0,2,4,6],
                'h': [1,10,100,1000],
                'i': [20,40,60,80],
                'j': [1e0, 1e4, 1e8, 1e12, 1e16]
            },
            1: {
                'a': ['a', 'b', 'c', 'd'],
                'b': [1, 2, 3, 4],
                'c': [10, 20, 30, 40],
                'd': [1, 100, 10000],
                'e': [25, 50, 75],
                'f': [0, 0.5, 1.0],
                'g': [0, 2, 4, 6],
                'h': [1, 10, 100, 1000],
                'i': [20, 40, 60, 80],
                'j': [1e0, 1e4, 1e8, 1e12, 1e16]
            }
    }

@pytest.fixture
def best_params():
    return {
            'a': 'a',
            'b': 2,
            'c': 20,
            'd': 100,
            'e': 50,
            'f': 0.5,
            'g': 2,
            'h': 100,
            'i': 40,
            'j': 1e16
    }

@pytest.fixture
def new_params():
    return {
        'a': [['a', 'b', 'c', 'd'], 3, 'string'],
        'b': [[1, 2, 3, 4], [4, 4, 4], 'fixed_integer'],
        'c': [[10, 20, 30, 40], [4, 4, 4], 'fixed_float'],
        'd': [[1,100,10000], [3, 5, 3], 'soft_integer'],
        'e': [[25, 50, 75], [3, 3, 3], 'soft_integer'],
        'f': [[0, 0.5, 1.0], [3, 3, 3], 'hard_float'],
        'g': [[0, 2, 4, 6], [4, 4, 4], 'hard_integer'],
        'h': [[1, 10, 100, 1000], [4, 4, 4], 'soft_float'],
        'i': [[20, 40, 60, 80], [4, 4, 4], 'soft_float'],
        'j': [[1e0, 1e4, 1e8, 1e12, 1e16], [5, 9, 5], 'soft_float']
    }


@pytest.fixture
def new_is_logspace():
    return {
            'a': False,
            'b': False,
            'c': False,
            'd': 1,
            'e': False,
            'f': False,
            'g': False,
            'h': 1,
            'i': False,
            'j': 1
    }


@pytest.fixture
def new_grids():
    return {
            0: {
                'a': ['a', 'b', 'c', 'd'],
                'b': [1, 2, 3, 4],
                'c': [10, 20, 30, 40],
                'd': [1, 100, 10000],
                'e': [25, 50, 75],
                'f': [0, 0.5, 1.0],
                'g': [0, 2, 4, 6],
                'h': [1, 10, 100, 1000],
                'i': [20, 40, 60, 80],
                'j': [1e0, 1e4, 1e8, 1e12, 1e16]
            },
            1: {
                'a': ['a', 'b', 'c', 'd'],
                'b': [1, 2, 3, 4],
                'c': [10, 20, 30, 40],
                'd': [1e0, 1e1, 1e2, 1e3, 1e4],
                'e': [25, 50, 75],
                'f': [0, 0.5, 1.0],
                'g': [0, 2, 4, 6],
                'h': [1, 10, 100, 1000],
                'i': [20, 40, 60, 80],
                'j': [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]
            }
    }


class TestRegapLogspace:

    # no validation

    def test_accuracy(self, good_grids, good_is_logspace, good_params,
                  best_params, new_grids, new_is_logspace, new_params):

        out_grids, out_params, out_is_logspace = \
            _regap_logspace(
                _GRIDS=good_grids,
                _IS_LOGSPACE=good_is_logspace,
                _params=good_params,
                _pass=1,
                _best_params_from_previous_pass=best_params
            )

        assert out_grids == new_grids
        assert out_params == new_params
        assert out_is_logspace == new_is_logspace




















