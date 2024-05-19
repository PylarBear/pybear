# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid._update_phlite import _update_phlite


@pytest.fixture
def _params():
    return {
            'a': [[1, 2, 3, 4], [4,4,4], 'soft_integer'],
            'b': [[1e-2, 1e-1, 1e0, 1e1], [4,4,4], 'soft_float'],
            'c': [[1, 10, 100, 1000], [4,4,4], 'soft_float'],
            'd': [[2, 3, 4, 5], [4,4,4], 'fixed_integer'],
            'e': [[1.1, 1.2, 1.3, 1.4], [4,4,4], 'fixed_float'],
            'f': [[0, 0.25, 0.5, 0.75, 1.0], [5,5,5], 'hard_float'],
            'g': [[1, 10, 100, 1000], [4, 4, 4], 'soft_float'],
            'h': [[1, 2, 3, 4], [4, 4, 4], 'soft_integer'],
            'i': [[1, 2, 3, 4], [4, 4, 4], 'soft_integer'],
            'j': [[0, 0.25, 0.5, 0.75, 1.0], [5, 5, 5], 'hard_float'],
            'k': [[1, 10, 100, 1000], [4, 4, 4], 'soft_float'],
    }


@pytest.fixture
def param_grid():
    return {
            'a': [1,2,3,4],   # 1 is int hard bound
            'b': [1e-2, 1e-1, 1e0, 1e1],
            'c': [1, 10, 100, 1000],
            'd': [2, 3, 4, 5],
            'e': [1.1, 1.2, 1.3, 1.4],
            'f': [0, 0.25, 0.5, 0.75, 1.0],  # 0 is float hard bound
            'g': [1, 10, 100, 1000],
            'h': [1, 2, 3, 4],
            'i': [1, 2, 3, 4],
            'j': [0, 0.25, 0.5, 0.75, 1.0],
            'k': [1, 10, 100, 1000]
    }

@pytest.fixture
def _best_params():
    return {
            'a': [1],  # 1 is int hard bound  TRUE
            'b': [1e1],  # FALSE
            'c': [1000],  # FALSE
            'd': [5],  # TRUE
            'e': [1.1],  # TRUE
            'f': [0],   # 0 is float hard bound   # TRUE
            'g': [100], # TRUE
            'h': [2],  # TRUE
            'i': [4],  # FALSE
            'j': [0.5],  # TRUE
            'k': [1],  # FALSE
    }

@pytest.fixture
def start_phlite_1():
    return {
            'a': False,
            'b': False,
            'c': True,
            'g': False,
            'h': True,
            'i': False,
            'k': False
    }

@pytest.fixture
def start_phlite_2():
    return {
            'a': True,
            'b': True,
            'c': True,
            'g': True,
            'h': True,
            'i': True,
            'k': True
    }

@pytest.fixture
def start_phlite_3():
    return {
            'a': True,
            'b': False,
            'c': False,
            'g': True,
            'h': True,
            'i': False,
            'k': False
    }

@pytest.fixture
def start_phlite_4():
    return {
            'a': False,
            'b': True,
            'c': True,
            'g': False,
            'h': False,
            'i': True,
            'k': True
    }

@pytest.fixture
def start_phlite_5():
    return {
            'a': True,
            'b': False,
            'c': False,
            'g': True,
            'h': False,
            'i': False,
            'k': False
    }



@pytest.fixture
def final_phlite():
    return {
            'a': True,
            'b': False,
            'c': False,
            'g': True,
            'h': True,
            'i': False,
            'k': False
    }


class TestUpdatePhlite:

    def test_rejects_bad_phlite(self, _params, param_grid, _best_params,
                                start_phlite_1, final_phlite):

        with pytest.raises(ValueError):

            bad_phlite = start_phlite_1 | {'d':True, 'e':False, 'f':True,
                                           'j':False}

            _update_phlite(bad_phlite, param_grid, _params, _best_params)


    def test_accuracy_1(self, _params, param_grid, _best_params,
                        start_phlite_1, final_phlite):

        assert _update_phlite(start_phlite_1, param_grid, _params, _best_params)


    def test_accuracy_2(self, _params, param_grid, _best_params,
                        start_phlite_2, final_phlite):

        assert _update_phlite(start_phlite_2, param_grid, _params, _best_params)


    def test_accuracy_3(self, _params, param_grid, _best_params,
                        start_phlite_3, final_phlite):

        assert _update_phlite(start_phlite_3, param_grid, _params, _best_params)


    def test_accuracy_4(self, _params, param_grid, _best_params,
                        start_phlite_4, final_phlite):

        assert _update_phlite(start_phlite_4, param_grid, _params, _best_params)


    def test_accuracy_5(self, _params, param_grid, _best_params,
                        start_phlite_5, final_phlite):

        assert _update_phlite(start_phlite_5, param_grid, _params, _best_params)











