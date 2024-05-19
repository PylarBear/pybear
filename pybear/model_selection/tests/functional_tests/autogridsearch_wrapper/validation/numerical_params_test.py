# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _numerical_params import _numerical_param_value



@pytest.fixture
def good_num_3():
    return [[1,2,3,4], [4,4,4], 'hard_integer']


@pytest.fixture
def good_num_5():
    return ['linspace', 0, 10, [3,3,3], 'hard_integer']


class TestNumericalParamKey:

    @pytest.mark.parametrize('non_str',
    (0, 1, np.pi, True, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_reject_non_str(self, non_str):
        with pytest.raises(TypeError):
            _numerical_param_value(non_str, [['a','b'], 4, 'string'], 3)

    def test_accepts_str(self, good_num_3, good_num_5):
        assert _numerical_param_value('some_string', good_num_3, 3) == \
                [[1,2,3,4], [4,4,4], 'hard_integer']

        assert _numerical_param_value('some_string', good_num_5, 3) == \
                [[0, 5, 10], [3,3,3], 'hard_integer']




class TestNumericalParamValueOuterContainer:


    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key', non_list_like, 3)


    @pytest.mark.parametrize('list_like',
    (
     [[1,2,3], [3,3,3], 'soft_float'],
     ([1,2,3], [3,3,3], 'soft_float'),
     np.array([[1,2,3], [3,3,3], 'soft_float'], dtype=object)
     )
    )
    def test_accepts_list_like1(self, list_like):
        assert _numerical_param_value('good_key', list_like, 3) == \
               [[1,2,3], [3,3,3], 'soft_float']


    @pytest.mark.parametrize('list_like',
    (
     ['linspace', 1, 3, [3,3,3], 'fixed_integer'],
     ('linspace', 1, 3, [3,3,3], 'fixed_integer'),
     np.array(['linspace', 1, 3,  [3,3,3], 'fixed_integer'], dtype=object)
     )
    )
    def test_accepts_list_like1(self, list_like):
        assert _numerical_param_value('good_key', list_like, 3) == \
               [[1,2,3], [3,3,3], 'fixed_integer']


class TestSearchValues_ListOfValues:

    @pytest.mark.parametrize('non_list_like',
        (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                                   [non_list_like, [1], 'soft_float'], 1)


    def test_type_error_for_set(self):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key', [{1,2,3}, [3], 'hard_integer'], 1)


    @pytest.mark.parametrize('list_like',
         ([1,2,3], (1,2,3), np.array([1,2,3], dtype=object))
    )
    def test_accepts_list_like(self, list_like):
        assert _numerical_param_value('good_key',
                  [list_like, [3], 'hard_integer'], 1) == \
                                        [[1,2,3], [3], 'hard_integer']


    @pytest.mark.parametrize('non_str_non_none',
    (True, min, 'junk', lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_numeric_inside(self, non_str_non_none):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                            [[non_str_non_none, 2, 3], [3], 'soft_float'], 1)

    def test_integer_dtype_rejects_float(self):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                                   [[1,2,np.pi], [3,3,3], 'hard_integer'], 3)

    @pytest.mark.parametrize('_dtype',
        ('soft_integer, hard_integer, fixed_integer')
    )
    def test_int_rejects_lt_one(self, _dtype):

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                                [[0,1,2], [3,3,3], _dtype], 3)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                                [[1e-6,1e-6,1e-4], [3,3,3], _dtype], 3)


    @pytest.mark.parametrize('_dtype',
        ('soft_float, hard_float, fixed_float')
    )
    def test_float_rejects_lt_zero(self, _dtype):

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                            ['linspace', -1, 2, [3,3,3], _dtype], 3)


    @pytest.mark.parametrize('value', (0, 1, np.pi))
    def test_float_dtype_accepts_any_other_number(self, value):
        assert _numerical_param_value('good_key',
            [[1,2,value], [3,3,3], 'hard_float'], 3) == \
               [sorted([1.0,2.0,value]), [3,3,3], 'hard_float']


class TestSearchValues_Space:

    @pytest.mark.parametrize('non_str',
    (0, np.pi, True, None, min, [1], (1,), {1}, lambda x: x, {'a': 1})
    )
    def test_posn0_rejects_non_str(self, non_str):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                                   [non_str, 1, 10, [10], 'soft_float'], 1)


    @pytest.mark.parametrize('bad_str',
    ('junk', 'more_junk', 'and', 'even_more_junk')
    )
    def test_posn0_rejects_bad_str(self, bad_str):
        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                                   [bad_str, 1, 10, [10], 'soft_float'], 1)


    def test_posn0_accepts_good_str_1(self):
        assert _numerical_param_value('good_key',
           ['linspace', 1, 3, [3], 'soft_float'], 1) == [[1,2,3], [3], 'soft_float']


    def test_posn0_accepts_good_str_2(self):
        assert _numerical_param_value('good_key',
           ['logspace', 0, 2, [3], 'soft_integer'], 1) == \
               [[1, 10,100], [3], 'soft_integer']


    @pytest.mark.parametrize('non_numeric',
    ('junk', True, None, min, [1], (1,), {1}, lambda x: x, {'a': 1})
    )
    def test_posn1_posn2_rejects_non_numeric(self, non_numeric):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                        ['linspace', non_numeric, 10, [10], 'soft_float'], 1)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                        ['linspace', 0, non_numeric, [10], 'soft_float'], 1)


    def test_posn1_posn2_integer_dtype_rejects_float(self):

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                        ['linspace', 0 ,np.pi, [3,3,3], 'hard_integer'], 3)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                         ['linspace', np.pi, 10, [3,3,3], 'hard_integer'], 3)


    def test_posn1_posn2_float_accepts_any_number(self):
        assert _numerical_param_value('good_key',
            ['linspace', -2.5, -1.5, [3,3,3], 'hard_float'], 3) == \
               [[-2.5, -2.0, -1.5], [3,3,3], 'hard_float']


    @pytest.mark.parametrize('_dtype',
        ('soft_integer, hard_integer, fixed_integer')
    )
    def test_int_rejects_lt_one(self, _dtype):

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                            ['linspace', 0, 2, [3,3,3], _dtype], 3)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                            ['logspace', -4, 4, [9,9,9], _dtype], 3)


    @pytest.mark.parametrize('_dtype',
        ('soft_float, hard_float, fixed_float')
    )
    def test_float_rejects_lt_zero(self, _dtype):

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                            ['linspace', -1, 2, [3,3,3], _dtype], 3)


    def test_accuracy(self):
        assert _numerical_param_value('good_key',
              ['linspace', 1, 2, [2], 'hard_integer'], 1) == \
                    [[1, 2], [2], 'hard_integer']

        assert _numerical_param_value('good_key',
              ['logspace', 1, 3, 3, 'hard_integer'], 3) == \
                    [[10, 100, 1000], [3, 3, 3], 'hard_integer']


class TestPoints:

    @pytest.mark.parametrize('non_list_type',
    (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, {1,2})
    )
    def test_rejects_non_int_non_list_type(self, non_list_type):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                ['linspace', 1, 10, non_list_type, 'soft_integer'], 1)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                    [[1,2,3], non_list_type, 'soft_integer'], 1)


    @pytest.mark.parametrize('list_type',
                ([2,2], (2,2), np.array([2,2], dtype=int))
    )
    def test_accepts_list_type(self, list_type):
        assert _numerical_param_value('good_key',
                ['linspace', 1, 2, list_type, 'hard_integer'], 2) == \
                    [[1,2], [2,2], 'hard_integer']

        assert _numerical_param_value('good_key',
                [[1,2], list_type, 'hard_integer'], 2) == \
                    [[1,2], [2,2], 'hard_integer']


    def test_accepts_integer_gte_one(self):
        # THIS ALSO VALIDATES THAT SETTING passes TO None SETS PASSES TO
        # ONE MILLION
        assert _numerical_param_value('good_key',
                    [[1, 2], 2, 'fixed_integer'], 1) == \
                            [[1, 2], [2], 'fixed_integer']

        assert _numerical_param_value('good_key',
                    ['linspace', 1, 2, 2, 'fixed_integer'], 1) == \
                            [[1, 2], [2], 'fixed_integer']


    def test_rejects_none(self):
        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                                   [[1, 2], [2, None], 'fixed_integer'], 3)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                            ['linspace', 1, 2, [2, None], 'fixed_integer'], 3)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                                   [[1, 2], None, 'fixed_integer'], 3)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                            ['linspace', 1, 2, None, 'fixed_integer'], 3)


    def test_rejects_integer_less_than_one(self):
        with pytest.raises(ValueError):
            _numerical_param_value('good_key', [[1,2], -1, 'fixed_integer'], 1)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                                   ['linspace', 1, 2, -1, 'fixed_integer'], 1)


class TestArgType:

    @pytest.mark.parametrize('bad_param_type',
    (0, np.pi, True, None, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    def test_rejects_any_non_string(self, bad_param_type):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key', [['a','b'], None, bad_param_type], 2)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                       ['linspace', 1, 2, None, bad_param_type], 2)


    @pytest.mark.parametrize('bad_string', ('junk', 'and', 'more_junk'))
    def test_rejects_bad_strings(self, bad_string):
        with pytest.raises(ValueError):
            _numerical_param_value('good_key', [['a','b'], None, bad_string], 2)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                           ['linspace', 1, 2, None, bad_string], 2)


    @pytest.mark.parametrize('good_type',
    ('hard_float', 'hard_integer', 'soft_float', 'soft_integer', 'fixed_float',
     'fixed_integer')
    )
    def test_accepts_valid_strings(self, good_type):
        _numerical_param_value('good_key', [[1, 2, 3], 3, good_type], 1)
        _numerical_param_value('good_key', ['linspace', 1, 3, [3], good_type], 1)











