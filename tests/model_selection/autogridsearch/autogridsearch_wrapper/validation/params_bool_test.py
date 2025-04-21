# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _params_bool import _val_bool_param_value



class TestBoolParamKey:

    @pytest.mark.parametrize('non_str',
        (0, 1, np.pi, True, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_reject_non_str(self, non_str):
        with pytest.raises(TypeError):
            _val_bool_param_value(non_str, [[True, False], 4, 'bool'])


    def test_accepts_str(self):
        assert _val_bool_param_value(
            'some_string', [[True, False], 8, 'bool']
        ) is None


class TestBoolParamValueOuterContainer:


    @pytest.mark.parametrize('non_list_like',
        (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _val_bool_param_value('good_key', non_list_like)


    @pytest.mark.parametrize('_container', (list, tuple, np.ndarray))
    def test_accepts_list_like(self, _container):
        _base = [[True, False, None], 10, 'bool']
        if _container in [list, tuple]:
            list_like = _container(_base)
        elif _container is np.ndarray:
            list_like = np.array(_base, dtype=object)
        else:
            raise Exception

        assert isinstance(list_like, _container)
        assert _val_bool_param_value('good_key', list_like) is None



class TestBoolListOfSearchPoints:

    @pytest.mark.parametrize('non_list_like',
        (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _val_bool_param_value(
                'good_key',
                [non_list_like, None, 'bool'],
                _shrink_pass_can_be_None=True
            )


    @pytest.mark.parametrize('_container', (list, tuple, set, np.ndarray))
    def test_accepts_list_like(self, _container):
        _base = [True, False, None]
        if _container in [list, tuple, set]:
            _grid = _container(_base)
        elif _container is np.ndarray:
            _grid = np.array(_base, dtype=object)
        else:
            raise Exception

        _value = [_grid, 10, 'bool']
        assert _val_bool_param_value('good_key', [_grid, 10, 'bool']) is None


    @pytest.mark.parametrize('non_bool',
        (0, 2.7, np.pi, min, 'trash', lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_bool_non_None_in_grid(self, non_bool):
        with pytest.raises(TypeError):
            _val_bool_param_value(
                'good_key',
                [[non_bool, False], None, 'bool'],
                _shrink_pass_can_be_None=True
            )


    @pytest.mark.parametrize('_bool', (True, False, None))
    def test_accepts_bool_None_inside(self, _bool):
        assert _val_bool_param_value(
            'good_key', ([_bool, False], 5, 'bool')
        ) is None


class TestShrinkPass:

    @pytest.mark.parametrize('non_integer',
        (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_non_none_non_integer(self, non_integer):
        with pytest.raises(TypeError):
            _val_bool_param_value(
                'good_key',
                [[True, False], non_integer, 'bool']
            )


    @pytest.mark.parametrize('int_or_none', (3, None))
    @pytest.mark.parametrize('can_be_None', (True, False))
    def test_accepts_none_and_integer_gte_one(
        self, int_or_none, can_be_None
    ):

        if int_or_none is None and not can_be_None:
            with pytest.raises(TypeError):
                _val_bool_param_value(
                    'good_key',
                    [[True, False], int_or_none, 'bool'],
                    _shrink_pass_can_be_None=can_be_None
                )
        else:
            assert _val_bool_param_value(
                'good_key',
                [[True,False], int_or_none, 'bool'],
                _shrink_pass_can_be_None=can_be_None
            ) is None


    @pytest.mark.parametrize('bad_pass', (-1, 0, 1))
    def test_rejects_integer_less_than_two(self, bad_pass):
        with pytest.raises(ValueError):
            _val_bool_param_value(
                'good_key',
                [[True, False], bad_pass, 'bool']
            )


class TestParamType:

    @pytest.mark.parametrize('bad_param_type',
        (0, np.pi, True, None, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_anything_not_the_word_string(self, bad_param_type):
        with pytest.raises(TypeError):
            _val_bool_param_value(
                'good_key',
                [[True, False], None, bad_param_type],
                _shrink_pass_can_be_None=True
            )


    @pytest.mark.parametrize('bad_string', ('junk', 'and', 'more_junk'))
    def test_rejects_bad_strings(self, bad_string):
        with pytest.raises(ValueError):
            _val_bool_param_value(
                'good_key',
                [[True, False], None, bad_string],
                _shrink_pass_can_be_None=True
            )


    def test_accepts_the_word_bool(self):
        assert _val_bool_param_value(
            'good_key',
            [[True, False], 3, 'bool'],
        ) is None





