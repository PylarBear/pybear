# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _params_string import _val_string_param_value











class TestStringParamKey:


    def test_accepts_str(self):
        assert _val_string_param_value(
            'some_string', [['a','b'], 8, 'string']
        ) is None


class TestStringParamValueOuterContainer:


    @pytest.mark.parametrize('_container', (list, tuple, np.ndarray))
    def test_accepts_list_like(self, _container):
        _base = [['a', 'b'], 10, 'string']
        if _container in [list, tuple]:
            list_like = _container(_base)
        elif _container is np.ndarray:
            list_like = np.array(_base, dtype=object)
        else:
            raise Exception

        assert isinstance(list_like, _container)
        assert _val_string_param_value('good_key', list_like) is None


class TestStringListOfSearchPoints:


    @pytest.mark.parametrize('non_str_non_none',
        (0, np.pi, True, min, lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_strings_non_none_in_grid(self, non_str_non_none):
        with pytest.raises(TypeError):
            _val_string_param_value(
                'good_key',
                [[non_str_non_none, 'b'], None, 'string'],
                _shrink_pass_can_be_None=True
            )


    @pytest.mark.parametrize('str_or_none', ('a', None))
    def test_accept_strings_or_none_inside(self, str_or_none):
        assert _val_string_param_value(
            'good_key', ((str_or_none, 'b'), 5, 'string')
        ) is None


class TestShrinkPass:

    @pytest.mark.parametrize('non_integer',
        (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_non_none_non_integer(self, non_integer):
        with pytest.raises(TypeError):
            _val_string_param_value(
                'good_key',
                [['a','b'], non_integer, 'string']
            )


    @pytest.mark.parametrize('int_or_none', (3, None))
    @pytest.mark.parametrize('can_be_None', (True, False))
    def test_accepts_none_and_integer_gte_one(
        self, int_or_none, can_be_None
    ):

        if int_or_none is None and not can_be_None:
            with pytest.raises(TypeError):
                _val_string_param_value(
                    'good_key',
                    [['y', 'z'], int_or_none, 'string'],
                    _shrink_pass_can_be_None=can_be_None
                )
        else:
            assert _val_string_param_value(
                'good_key',
                [['y','z'], int_or_none, 'string'],
                _shrink_pass_can_be_None=can_be_None
            ) is None


    @pytest.mark.parametrize('bad_pass', (-1, 0, 1))
    def test_rejects_integer_less_than_two(self, bad_pass):
        with pytest.raises(ValueError):
            _val_string_param_value(
                'good_key',
                [['a','b'], bad_pass, 'string']
            )






