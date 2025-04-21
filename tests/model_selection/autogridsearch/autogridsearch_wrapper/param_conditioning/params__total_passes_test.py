# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np
from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._params__total_passes import _cond_params__total_passes




class TestParamsTotalPasses_Accuracy:


    @staticmethod
    @pytest.fixture
    def good_dict_params():
        return {
            'param_a': [['a', 'b', 'c'], None, 'string'],
            'param_b': [np.logspace(1, 3, 3), [3, 11, 6], 'soft_float'],
            'param_c': [[True, False], 2, 'bool']
        }


    @staticmethod
    @pytest.fixture
    def _inf_shrink_pass():
        return 800_000


    @staticmethod
    @pytest.fixture
    def answer_good_dict_params(_inf_shrink_pass):
        return {
            'param_a': [['a', 'b', 'c'], _inf_shrink_pass, 'string'],
            'param_b': [[10.0, 100.0, 1000.0], [3, 11, 6], 'soft_float'],
            'param_c': [[True, False], 2, 'bool']
        }


    # points len == passes from kwarg returns the same
    def test_same_returns_same(
        self, good_dict_params, answer_good_dict_params, _inf_shrink_pass
    ):

        _params_out, _passes_out = \
            _cond_params__total_passes(good_dict_params, 3, _inf_shrink_pass)

        assert _params_out == answer_good_dict_params
        assert _passes_out == 3


    # a list-type passed to any 'points' overrides kwarg passes
    @pytest.mark.parametrize('kwarg_passes', (1, 2, 4, 5))
    def test_overrides_kwarg_passes(
        self, kwarg_passes, good_dict_params, answer_good_dict_params, _inf_shrink_pass
    ):

        _params_out, _passes_out = _cond_params__total_passes(
            good_dict_params, kwarg_passes, _inf_shrink_pass
        )

        assert _params_out == answer_good_dict_params
        assert _passes_out == 3


    # when str param only, kwarg total_passes is always returned
    @pytest.mark.parametrize('kwarg_passes', (1,2,3,4,5))
    def test_str_bool_params_only(self, kwarg_passes, _inf_shrink_pass):

        _params = {
            'a': [['aa', 'bb', 'cc'], 2, 'string'],
            'b': [['dd', 'ee', 'ff'], 3, 'string'],
            'c': [['gg', 'hh', 'ii'], 4, 'string'],
            'd': [[True, False], 4, 'bool']
        }

        _params_out, _passes_out = \
            _cond_params__total_passes(_params, kwarg_passes, _inf_shrink_pass)

        assert _params_out == _params
        assert _passes_out == kwarg_passes


    # one list of points sets points for them all
    @pytest.mark.parametrize('kwarg_passes', (1,2,3,4,5))
    def test_propagation_of_total_passes(self, kwarg_passes, _inf_shrink_pass):

        _params = {
            'a': [np.logspace(-4, 4, 5), 5, 'soft_float'],
            'b': [np.linspace(100, 500, 5), 4, 'soft_integer'],
            'c': [[2, 3, 4, 5], [4,4,4,4], 'fixed_integer'],
            'd': [[True, False], 5, 'bool']
        }

        answer_params = {
            'a': [[1e-4, 1e-2, 1, 1e2, 1e4], [5,5,5,5], 'soft_float'],
            'b': [[100, 200, 300, 400, 500], [5,4,4,4], 'soft_integer'],
            'c': [[2, 3, 4, 5], [4,4,4,4], 'fixed_integer'],
            'd': [[True, False], 5, 'bool']
        }

        _params_out, _passes_out = \
            _cond_params__total_passes(_params, kwarg_passes, _inf_shrink_pass)

        assert _params_out == answer_params
        assert _passes_out == 4




















