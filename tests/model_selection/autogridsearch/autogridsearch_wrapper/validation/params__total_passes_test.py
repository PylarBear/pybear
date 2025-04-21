# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np
from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _params__total_passes import _val_params__total_passes




@pytest.fixture
def good_dict_params():
    return {
            'param_a': [['a', 'b', 'c'], None, 'string'],
            'param_b': [np.logspace(1, 3, 3), [3,11,6], 'soft_float'],
            'param_c': [[True, False], 2, 'bool']
    }



class TestParamsTotalPasses_Validation:

    # total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    #   when no param in params has points that are list-type, total_passes
    #   must be determined externally. 'string' and 'bool' never allows a list-type
    #   for points, all other numerics' points could be passed as single
    #   int or a list-like[int].

    @pytest.mark.parametrize('non_numeric',
        (True, False, None, 'string', [1,2], (1,2), {1,2}, lambda x: x, {'a':1})
    )
    def test_rejects_non_numeric(self, good_dict_params, non_numeric):
        with pytest.raises(TypeError):
            _val_params__total_passes(good_dict_params, non_numeric)


    @pytest.mark.parametrize('non_numeric',
        (-np.pi, -np.e, np.e, np.pi, float('inf'))
    )
    def test_rejects_non_integer(self, good_dict_params, non_numeric):
        with pytest.raises(TypeError):
            _val_params__total_passes(good_dict_params, non_numeric)


    @pytest.mark.parametrize('less_than_one', (0, -1))
    def test_rejects_less_than_one(self, good_dict_params, less_than_one):
        with pytest.raises(ValueError):
            _val_params__total_passes(good_dict_params, less_than_one)


    def test_accepts_good_positive_integer(self, good_dict_params):

        assert _val_params__total_passes(good_dict_params, 3) is None

    # END total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('non_iterable',
        (0, 1, True, None, np.pi, min, lambda x: x)
    )
    def test_rejects_non_iterable(self, non_iterable):
        with pytest.raises(TypeError):
            _val_params__total_passes(non_iterable, 3)


    @pytest.mark.parametrize('non_dict',
        ('junk', [1, 2], [[1, 2]], (1, 2), {1, 2}, np.array([1, 2], dtype=int))
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _val_params__total_passes(non_dict, 3)


    @pytest.mark.parametrize('non_str',
        (0, 1, True, None, np.pi, min, {1, 2}, [1, 2], (1, 2), lambda x: x)
    )
    def test_reject_non_str_keys(self, non_str):
        with pytest.raises(TypeError):
            _val_params__total_passes(
                {non_str: [[1, 2, 3], [3, 3, 3], 'hard_integer']},
                3
            )

        with pytest.raises(TypeError):
            _val_params__total_passes(
                {non_str: [[1, 2, 3], [3, 3, 3], 'hard_integer'],
                 'b': [[1.1, 2.1, 3.1], [3, 3, 3], 'hard_float']},
                3
            )


    @pytest.mark.parametrize('junk_dict',
        ({'a': 1}, {'a': 'junk'}, {'a': {1, 2, 3}}, {'a': None})
    )
    def test_rejects_junk_dictionaries(self, junk_dict):
        with pytest.raises((TypeError, ValueError)):
            _val_params__total_passes(junk_dict, 3)


    def test_accepts_dict(self, good_dict_params):
        assert _val_params__total_passes(good_dict_params, 3) is None


    def test_rejects_diff_points_len(self):

        _params = {
            'a': [[2,3,4,5], [4,4,4], 'fixed_integer'],
            'b': [np.logspace(0, 4), [5,5,5,5], 'soft_float']
        }

        with pytest.raises(ValueError):
            _val_params__total_passes(_params, 3)



    # pizza maybe move this to _val_numerical
    @pytest.mark.parametrize('_points', (4, 6))
    def test_rejects_non_int_log_gaps(self, _points):

        # also implies log gaps less than 1 are rejected

        _params = {
            'a': [np.logspace(-4, 4, _points), _points, 'soft_float'],
            'b': [np.linspace(100, 500, 5), 4, 'soft_integer'],
            'c': [[2, 3, 4, 5], [4, 4, 4, 4], 'fixed_integer']
        }

        with pytest.raises(ValueError):
            _val_params__total_passes(_params, 4)


    # END _params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **









