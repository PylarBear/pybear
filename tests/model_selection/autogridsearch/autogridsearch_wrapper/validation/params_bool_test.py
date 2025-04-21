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


    def test_accepts_str(self):
        assert _val_bool_param_value(
            'some_string', [[True, False], 8, 'bool']
        ) is None


class TestBoolParamValueOuterContainer:


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




