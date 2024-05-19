# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from model_selection.autogridsearch._autogridsearch_wrapper._validation._params \
    import _params_validation


@pytest.fixture
def good_list_params():
    return [
        {
            'param_a': [['a'], 1, 'string'],
            'param_b': [[1, 2, 3], [3, 3, 3], 'hard_integer']
        },
        {
            'param_a': [['b'], 1, 'string'],
            'param_b': [[4, 5, 6], [3, 3, 3], 'hard_integer']
        },
    ]


@pytest.fixture
def good_dict_params():
    return {
            'param_a': [['a', 'b', 'c'], None, 'string'],
            'param_b': ['logspace', 1, 3, [3,11,6], 'soft_float']
    }

@pytest.fixture
def answer_good_dict_params():
    return [{
            'param_a': [['a', 'b', 'c'], 1_000_000, 'string'],
            'param_b': [[10.0, 100.0, 1000.0], [3,11,6], 'soft_float']
    }]



class TestParamsValidation:

    @pytest.mark.parametrize('non_list_non_dict',
         ('junk', 0, 1, True, None, np.pi, min, {1,2}, lambda x: x)
    )
    def test_rejects_non_list_non_dict(self, non_list_non_dict):

        with pytest.raises(TypeError):
            _params_validation(non_list_non_dict, 3)


    def test_accepts_list_and_dict(self, good_list_params, good_dict_params,
                                   answer_good_dict_params):

        assert _params_validation(good_list_params, 3) == good_list_params

        assert _params_validation(good_dict_params, 3) == answer_good_dict_params


    @pytest.mark.parametrize('non_int',
        ('junk', True, None, np.pi, min, {1, 2}, lambda x: x)
    )
    def test_rejects_non_integer_total_passes(self, good_dict_params, non_int):
        with pytest.raises(TypeError):
            _params_validation(good_dict_params, non_int)


    @pytest.mark.parametrize('non_str',
        (0, 1, True, None, np.pi, min, {1, 2}, [1,2], (1,2), lambda x: x)
    )
    def test_reject_non_str_keys(self, non_str):

        with pytest.raises(TypeError):
            _params_validation(
                { non_str: [[1,2,3], [3,3,3], 'hard_integer']},
                3
            )


        with pytest.raises(TypeError):
            _params_validation(
                [
                {non_str: [[1,2,3], [3,3,3], 'hard_integer']},
                {'b': [[1.1, 2.1, 3.1], [3, 3, 3], 'hard_float']},
                ],
                3
            )

    @pytest.mark.parametrize('junk_dict',
    ({'a':1}, {'a': 'junk'}, {'a':{1,2,3}}, {'a': None})
    )
    def test_rejects_junk_dictionaries(self, junk_dict):
        with pytest.raises((TypeError, ValueError)):
            _params_validation(junk_dict, 3)

























