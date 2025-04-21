# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np
from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _validation._params import _val_params



class TestParamsValidation:


    @staticmethod
    @pytest.fixture
    def good_dict_params():
        return {
            'param_a': [['a', 'b', 'c'], None, 'string'],
            'param_b': [np.logspace(1, 3, 3), [3, 11, 6], 'soft_float'],
            'param_c': [[True, False], 2, 'bool']
        }


    # _params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('non_iterable',
        (0, 1, True, None, np.pi, min, lambda x: x)
    )
    def test_rejects_non_iterable(self, non_iterable):
        with pytest.raises(TypeError):
            _val_params(non_iterable, 3)


    @pytest.mark.parametrize('non_dict',
        ('junk', [1, 2], [[1, 2]], (1, 2), {1, 2}, np.array([1, 2], dtype=int))
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _val_params(non_dict, 3)


    @pytest.mark.parametrize('non_str',
        (0, 1, True, None, np.pi, min, {1, 2}, [1, 2], (1, 2), lambda x: x)
    )
    def test_reject_non_str_keys(self, non_str):
        with pytest.raises(TypeError):
            _val_params(
                {non_str: [[1, 2, 3], [3, 3, 3], 'hard_integer']},
                3
            )

        with pytest.raises(TypeError):
            _val_params(
                {non_str: [[1, 2, 3], [3, 3, 3], 'hard_integer'],
                 'b': [[1.1, 2.1, 3.1], [3, 3, 3], 'hard_float']},
                3
            )


    @pytest.mark.parametrize('junk_dict',
        ({'a': 1}, {'a': 'junk'}, {'a': {1, 2, 3}}, {'a': None})
    )
    def test_rejects_junk_dictionaries(self, junk_dict):
        with pytest.raises((TypeError, ValueError)):
            _val_params(junk_dict, 3)


    def test_accepts_dict(self, good_dict_params):
        assert _val_params(good_dict_params, 3) is None


    @pytest.mark.parametrize('total_passes', (2, 4))
    def test_rejects_bad_len(self, total_passes):

        with pytest.raises(ValueError):
            _val_params(
                {'good_key': [[1, 2, 3], [3, 3, 3]]}, total_passes
            )


@pytest.mark.parametrize('_type',
    ['fixed_integer', 'fixed_float', 'hard_integer',
    'hard_float', 'soft_integer', 'soft_float']
)
@pytest.mark.parametrize('total_passes', (1, 3))
class TestFirstGrid:


    @pytest.mark.parametrize('non_list_like',
        (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(
        self, non_list_like, total_passes, _type
    ):
        with pytest.raises(TypeError):
            _val_params(
                'good_key',
                [non_list_like, [1 for _ in range(total_passes)], _type],
                total_passes
            )


    @pytest.mark.parametrize('list_like',
         ([1,2,3], (1,2,3), {1,2,3}, np.array([1,2,3], dtype=object))
    )
    def test_accepts_list_like(self, list_like, total_passes, _type):

        points = [3 for _ in range(total_passes)]

        assert _val_params(
            {'good_key': [list_like, points, _type]},
            total_passes
        ) is None


    def test_rejects_empty(self, total_passes, _type):

        with pytest.raises(ValueError):
            _val_params(
                {'good_key': [[], [2 for _ in range(total_passes)], _type]},
                total_passes
            )


class TestOtherpizza:


    def test_rejects_diff_points_len(self):

        _params = {
            'a': [[2,3,4,5], [4,4,4], 'fixed_integer'],
            'b': [np.logspace(0, 4), [5,5,5,5], 'soft_float']
        }

        with pytest.raises(ValueError):
            _val_params(_params, 3)

    # END _params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **




class TestType:


    @pytest.mark.parametrize('bad_param_type',
        (0, np.pi, True, None, min, lambda x: x, {'a': 1}, [1, ], (1,), {1, 2})
    )
    def test_rejects_any_non_string(self, bad_param_type):
        with pytest.raises(TypeError):
            _val_params(
                'good_key', [['a', 'b'], None, bad_param_type], 2
            )


    @pytest.mark.parametrize('bad_string', ('junk', 'and', 'more_junk'))
    def test_rejects_bad_strings(self, bad_string):
        with pytest.raises(ValueError):
            _val_params(
                {'good_key': [['a', 'b'], None, bad_string]}, 2
            )


    @pytest.mark.parametrize('good_type',
        ['fixed_integer', 'fixed_float', 'hard_integer',
        'hard_float', 'soft_integer', 'soft_float']
    )
    def test_accepts_valid_strings(self, good_type):
        assert _val_params(
            {'good_key': [[1, 2, 3], 3, good_type]}, 1
        ) is None






