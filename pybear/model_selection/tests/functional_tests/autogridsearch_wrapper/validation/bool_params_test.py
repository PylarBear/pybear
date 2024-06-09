# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _bool_params import _bool_param_value




class TestBoolParamKey:

    @pytest.mark.parametrize('non_str',
    (0, 1, np.pi, True, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_reject_non_str(self, non_str):
        with pytest.raises(TypeError):
            _bool_param_value(non_str, [[True, False], 4, 'bool'])

    def test_accepts_str(self):
        assert _bool_param_value('some_string', [[True, False], 8, 'bool']) == \
                [[True, False], 8, 'bool']




class TestBoolParamValueOuterContainer:


    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _bool_param_value('good_key', non_list_like)


    @pytest.mark.parametrize('list_like',
    (
     [[True, False], 10, 'bool'],
     ([True, False], 10, 'bool'),
     np.array([(True, False), 10, 'bool'], dtype=object)
     )
    )
    def test_accepts_list_like(self, list_like):
        assert _bool_param_value('good_key', list_like) == \
               [[True, False], 10, 'bool']



class TestListOfArgs:

    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _bool_param_value('good_key', [non_list_like, None, 'bool'])


    @pytest.mark.parametrize('list_like',
         ([True, False], (True, False), np.array([True, False], dtype=object))
    )
    def test_accepts_list_like(self, list_like):
        assert _bool_param_value('good_key', [list_like, 10, 'bool']) == \
               [[True, False], 10, 'bool']


    @pytest.mark.parametrize('non_bool',
    (0, np.pi, None, min, lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_bool_inside(self, non_bool):
        with pytest.raises(TypeError):
            _bool_param_value('good_key', [[non_bool, False], None, 'bool'])


    @pytest.mark.parametrize('_bool', (True, False))
    def test_accepts_bool_inside(self, _bool):
        assert _bool_param_value('good_key', ([_bool, False], 5, 'bool')) == \
               [[_bool, False], 5, 'bool']



class TestPasses:

    @pytest.mark.parametrize('non_none_non_integer',
    (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_non_none_non_integer(self, non_none_non_integer):
        with pytest.raises(TypeError):
            _bool_param_value('good_key',
                    [[True, False], non_none_non_integer, 'bool'])


    @pytest.mark.parametrize('int_or_none', (3, None))
    def test_accepts_none_and_integer_gte_one(self, int_or_none):
        # THIS ALSO VALIDATES THAT SETTING passes TO None SETS PASSES TO
        # ONE MILLION
        assert _bool_param_value('good_key', [[True,False], int_or_none, 'bool']) == \
            [[True, False], int_or_none or 1_000_000, 'bool']


    @pytest.mark.parametrize('bad_pass', (-1, 0, 1))
    def test_rejects_integer_less_than_two(self, bad_pass):
        with pytest.raises(ValueError):
            _bool_param_value('good_key', [[True, False], bad_pass, 'bool'])


class TestArgType:

    @pytest.mark.parametrize('bad_param_type',
    (0, np.pi, True, None, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_anything_not_the_word_string(self, bad_param_type):
        with pytest.raises(TypeError):
            _bool_param_value('good_key', [[True, False], None, bad_param_type])


    @pytest.mark.parametrize('bad_string', ('junk', 'and', 'more_junk'))
    def test_rejects_bad_strings(self, bad_string):
        with pytest.raises(ValueError):
            _bool_param_value('good_key', [[True, False], None, bad_string])


    def test_accepts_the_word_bool(self):
        assert _bool_param_value('good_key', [[True, False], 3, 'bool']) == \
                        [[True, False], 3, 'bool']















