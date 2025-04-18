# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np
from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _params_string import _string_param_value



class TestStringParamKey:

    @pytest.mark.parametrize('non_str',
    (0, 1, np.pi, True, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_reject_non_str(self, non_str):
        with pytest.raises(TypeError):
            _string_param_value(non_str, [['a','b'], 4, 'string'])

    def test_accepts_str(self):
        assert _string_param_value('some_string', [['a','b'], 8, 'string']) == \
                [['a', 'b'], 8, 'string']




class TestStringParamValueOuterContainer:


    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _string_param_value('good_key', non_list_like)


    @pytest.mark.parametrize('list_like',
    (
     [['a', 'b'], 10, 'string'],
     (['a', 'b'], 10, 'string'),
     np.array([('a', 'b'), 10, 'string'], dtype=object)
     )
    )
    def test_accepts_list_like(self, list_like):
        assert _string_param_value('good_key', list_like) == \
               [['a', 'b'], 10, 'string']



class TestListOfArgs:

    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _string_param_value('good_key', [non_list_like, None, 'string'])


    @pytest.mark.parametrize('list_like',
         (['a', 'b'], ('a', 'b'), np.array(['a', 'b'], dtype=object))
    )
    def test_accepts_list_like(self, list_like):
        assert _string_param_value('good_key', [list_like, 10, 'string']) == \
               [['a', 'b'], 10, 'string']


    @pytest.mark.parametrize('non_str_non_none',
    (0, np.pi, True, min, lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_strings_non_none_inside(self, non_str_non_none):
        with pytest.raises(TypeError):
            _string_param_value('good_key',
                                [[non_str_non_none, 'b'], None, 'string'])


    @pytest.mark.parametrize('str_or_none', ('a', None))
    def test_accept_strings_or_none_inside(self, str_or_none):
        assert _string_param_value('good_key', ([str_or_none, 'b'], 5, 'string')) == \
               [[str_or_none, 'b'], 5, 'string']



class TestPasses:

    @pytest.mark.parametrize('non_none_non_integer',
    (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_non_none_non_integer(self, non_none_non_integer):
        with pytest.raises(TypeError):
            _string_param_value('good_key',
                        [['a','b'], non_none_non_integer, 'string'])


    @pytest.mark.parametrize('int_or_none', (3, None))
    def test_accepts_none_and_integer_gte_one(self, int_or_none):
        # THIS ALSO VALIDATES THAT SETTING passes TO None SETS PASSES TO
        # ONE MILLION
        assert _string_param_value('good_key', [['a','b'], int_or_none, 'string']) == \
            [['a', 'b'], int_or_none or 1_000_000, 'string']


    @pytest.mark.parametrize('bad_pass', (-1, 0, 1))
    def test_rejects_integer_less_than_two(self, bad_pass):
        with pytest.raises(ValueError):
            _string_param_value('good_key', [['a','b'], bad_pass, 'string'])


class TestArgType:

    @pytest.mark.parametrize('bad_param_type',
    (0, np.pi, True, None, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_anything_not_the_word_string(self, bad_param_type):
        with pytest.raises(TypeError):
            _string_param_value('good_key', [['a','b'], None, bad_param_type])


    @pytest.mark.parametrize('bad_string', ('junk', 'and', 'more_junk'))
    def test_rejects_bad_strings(self, bad_string):
        with pytest.raises(ValueError):
            _string_param_value('good_key', [['a','b'], None, bad_string])


    def test_accepts_the_word_string(self):
        assert _string_param_value('good_key', [['a','b'], 3, 'string']) == \
                        [['a','b'], 3, 'string']












