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
    return ['linspace', 1, 9, [3,3,3], 'hard_integer']


class TestNumericalParamKey:

    @pytest.mark.parametrize('non_str',
    (0, 1, np.pi, True, min, lambda x: x, {'a': 1}, [1,], (1,), {1,2})
    )
    @pytest.mark.parametrize('total_passes', (1,3))
    def test_reject_non_str(self, non_str, total_passes):
        with pytest.raises(TypeError):
            _numerical_param_value(non_str, [['a','b'], 4, 'string'], total_passes)

    @pytest.mark.parametrize('total_passes', (1,3))
    def test_accepts_str(self, good_num_3, good_num_5, total_passes):

        good_num_3[-2] = [good_num_3[-2][0] for _ in range(total_passes)]
        out = _numerical_param_value('some_string', good_num_3, total_passes)
        assert isinstance(out, list)
        assert out == [[1,2,3,4], good_num_3[-2], 'hard_integer']

        good_num_5[-2] = [good_num_5[-2][0] for _ in range(total_passes)]
        out = _numerical_param_value('some_string', good_num_5, total_passes)
        assert isinstance(out, list)
        assert out == [[1, 5, 9], good_num_5[-2], 'hard_integer']


class TestNumericalParamValueOuterContainer:

    @pytest.mark.parametrize('non_list_like',
    (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_rejects_non_list_like(self, non_list_like, total_passes):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key', non_list_like, total_passes)


    @pytest.mark.parametrize('total_passes', (1,3))
    @pytest.mark.parametrize('list_like', ('list', 'tuple', 'np.array'))
    def test_accepts_list_like1(self, list_like, total_passes):
        _ = [[1,2,3], [3 for _ in range(total_passes)], 'soft_float']
        if list_like == 'list': list_like = list(_)
        elif list_like == 'tuple': list_like = tuple(_)
        elif list_like == 'np.array': list_like = np.array(_, dtype=object)

        out = _numerical_param_value('good_key', list_like, total_passes)
        assert isinstance(out, list)
        assert out == list(_)


    @pytest.mark.parametrize('total_passes', (1,3))
    @pytest.mark.parametrize('list_like', ('list', 'tuple', 'np.array'))
    def test_accepts_list_like2(self, list_like, total_passes):
        _ = ['linspace', 1, 3, [3 for _ in range(total_passes)], 'fixed_integer']
        if list_like == 'list': list_like = list(_)
        elif list_like == 'tuple': list_like = tuple(_)
        elif list_like == 'np.array': list_like = np.array(_, dtype=object)

        out = _numerical_param_value('good_key', list_like, total_passes)
        assert isinstance(out, list)
        assert out == [[1,2,3]] + _[-2:]


class TestGridAsListOfValues:

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('non_list_like',
        (0, np.pi, True, None, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_list_like(self, non_list_like, total_passes):
        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key',
                [non_list_like, [1 for _ in range(total_passes)], 'soft_float'],
                total_passes
            )


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('list_like',
         ([1,2,3], (1,2,3), {1,2,3}, np.array([1,2,3], dtype=object))
    )
    def test_accepts_list_like(self, list_like, total_passes):

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value(
            'good_key', [list_like, points, 'hard_integer'], total_passes
        )

        assert isinstance(out, list)
        assert out == [[1,2,3], [3 for _ in range(total_passes)], 'hard_integer']


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('non_str_non_none',
    (min, 'junk', lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_numeric_inside(self, non_str_non_none, total_passes):
        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key',
                [
                 [non_str_non_none, 2, 3],
                 [3 for _ in range(total_passes)],
                 'soft_float'
                 ],
                total_passes
            )

    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_integer_dtype_rejects_float(self, total_passes):
        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key',
                [[1,2,np.pi], [3 for _ in range(total_passes)], 'hard_integer'],
                total_passes
            )


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_dtype', ('soft_integer, hard_integer, fixed_integer'))
    def test_int_rejects_lt_one(self, total_passes, _dtype):

        points = [3 for _ in range(total_passes)]

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[0,1,2], points, _dtype], total_passes)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[1e-6,1e-6,1e-4], points, _dtype], total_passes)


    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_int_dtype_rejects_bool(self, total_passes):

        points = [2 for _ in range(total_passes)]

        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key',
                [[True, False], points, 'fixed_integer'],
                total_passes
            )


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_dtype', ('soft_float, hard_float, fixed_float'))
    def test_float_rejects_lt_zero(self, total_passes, _dtype):

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key',
                ['linspace', -1, 2, [3 for _ in range(total_passes)], _dtype],
                total_passes
            )


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('value', (0, 1, np.pi))
    def test_float_dtype_accepts_any_other_number(self, value, total_passes):

        points =[3 for _ in range(total_passes)]

        out = _numerical_param_value(
            'good_key',
            [[1,2,value], points, 'hard_float'],
            total_passes
        )

        assert isinstance(out, list)
        assert out == [sorted([1.0, 2.0, value]), points, 'hard_float']


    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_float_dtype_rejects_bool(self, total_passes):

        points = [2 for _ in range(total_passes)]

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                [[True, False], points, 'fixed_float'], total_passes
            )



class TestGridAsNpSpace:

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('non_str',
    (0, np.pi, True, None, min, [1], (1,), {1}, lambda x: x, {'a': 1})
    )
    def test_posn0_rejects_non_str(self, total_passes, non_str):
        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key',
                [non_str, 1, 10, [10 for _ in range(total_passes)], 'soft_float'],
                total_passes
            )


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('bad_str',
        ('junk', 'more_junk', 'and', 'even_more_junk')
    )
    def test_posn0_rejects_bad_str(self, total_passes, bad_str):
        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                [bad_str, 1, 10, [10 for _ in range(total_passes)], 'soft_float'],
                total_passes
            )

    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_posn0_accepts_good_str_1(self, total_passes):

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value(
            'good_key',
            ['linspace', 1, 3, points, 'soft_float'],
            total_passes
        )

        assert isinstance(out, list)
        assert out == [[1,2,3], points, 'soft_float']


    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_posn0_accepts_good_str_2(self, total_passes):

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value(
            'good_key',
            ['logspace', 0, 2, points, 'soft_integer'],
            total_passes
        )

        assert isinstance(out, list)
        assert out == [[1, 10,100], points, 'soft_integer']

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('non_numeric',
    ('junk', True, None, min, [1], (1,), {1}, lambda x: x, {'a': 1})
    )
    def test_posn1_posn2_rejects_non_numeric(self, total_passes, non_numeric):

        points = [10 for _ in range(total_passes)]

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                ['linspace', non_numeric, 10, points, 'soft_float'], total_passes)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                ['linspace', 0, non_numeric, points, 'soft_float'], total_passes)

    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_posn1_posn2_integer_dtype_rejects_float(self, total_passes):

        points = [3 for _ in range(total_passes)]

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                ['linspace', 0 ,np.pi, points, 'hard_integer'], total_passes)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                ['linspace', np.pi, 10, points, 'hard_integer'], total_passes)

    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_posn1_posn2_float_accepts_any_number(self, total_passes):

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value(
            'good_key',
            ['linspace', 1.5, 2.5, points, 'hard_float'],
            total_passes
        )

        assert isinstance(out, list)
        assert out == [[1.5, 2.0, 2.5], points, 'hard_float']


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_dtype', ('soft_integer, hard_integer, fixed_integer'))
    def test_int_rejects_lt_one(self, total_passes, _dtype):

        points = [3 for _ in range(total_passes)]

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                ['linspace', 0, 2, points, _dtype],
                total_passes
            )

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                ['logspace', -4, 4, [9,9,9], _dtype],
                total_passes
            )

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_dtype', ('soft_float, hard_float, fixed_float'))
    def test_float_rejects_lt_zero(self, total_passes, _dtype):

        points = [3 for _ in range(total_passes)]

        with pytest.raises(ValueError):
            _numerical_param_value('good_key',
                ['linspace', -1, 2, points, _dtype],
                total_passes
            )

    @pytest.mark.parametrize('total_passes', (1, 3))
    def test_accuracy(self, total_passes):

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value('good_key',
            ['linspace', 1, 3, points, 'hard_integer'],
            total_passes
        )

        assert isinstance(out, list)
        assert out == [[1, 2, 3], points, 'hard_integer']

        out = _numerical_param_value('good_key',
              ['logspace', 1, 3, 3, 'hard_integer'], total_passes
        )

        assert isinstance(out, list)
        assert out == [[10, 100, 1000], points, 'hard_integer']

    @pytest.mark.parametrize('_type',
                         ('soft_integer', 'hard_integer', 'fixed_integer')
    )
    @pytest.mark.parametrize('_space', ('linspace', 'logspace'))
    @pytest.mark.parametrize('_points', (3, 9))
    def test_integer_rejects_non_unit_gaps(self, _type, _space, _points):
        with pytest.raises(TypeError):
            _numerical_param_value('a', [_space, 1, 6, _points, _type], 3)


class TestPoints:

    allowed_types = [
        'fixed_integer',
        'fixed_float',
        'hard_integer',
        'hard_float',
        'soft_integer',
        'soft_float'
    ]

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_type', allowed_types)
    @pytest.mark.parametrize('non_list_type',
        (np.pi, True, min, 'junk', lambda x: x, {'a': 1}, {1,2})
    )
    def test_rejects_non_int_non_list_type(self, total_passes, _type, non_list_type):
        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key', ['linspace', 1, 10, non_list_type, _type], total_passes
            )

        with pytest.raises(TypeError):
            _numerical_param_value(
                'good_key', [[1,2,3], non_list_type, _type], total_passes
            )

    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_type', allowed_types)
    @pytest.mark.parametrize('list_type', (list, tuple, np.array))
    def test_accepts_list_type(self, total_passes, list_type, _type):

        points = list_type([3 for _ in range(total_passes)])
        try:
            points = points.astype(int)
        except:
            pass

        out = _numerical_param_value(
            'good_key', [[1,2,3], points, _type], total_passes
        )

        assert isinstance(out, list)
        assert out == [[1,2,3], list(points), _type]

        out = _numerical_param_value(
            'good_key', ['linspace', 1, 3, points, _type], total_passes
        )

        assert isinstance(out, list)
        assert out == [[1,2,3], list(points), _type]


    @pytest.mark.parametrize('total_passes', (1, 3))
    @pytest.mark.parametrize('_type', allowed_types)
    def test_accepts_integer_gte_one(self, total_passes, _type):
        # THIS ALSO VALIDATES THAT SETTING passes TO None SETS PASSES TO
        # ONE MILLION

        points = [3 for _ in range(total_passes)]

        out = _numerical_param_value('good_key', [[1, 2, 3], 3, _type], total_passes)
        assert isinstance(out, list)
        assert out == [[1, 2, 3], points, _type]

        out = _numerical_param_value(
            'good_key', ['linspace', 1, 3, 3, _type], total_passes
        )
        assert isinstance(out, list)
        assert out == [[1, 2, 3], points, _type]


    @pytest.mark.parametrize('_type', allowed_types)
    def test_rejects_none(self, _type):
        with pytest.raises(TypeError):
            _numerical_param_value('good_key', [[1, 2], [2, None], _type], 2)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key',
                                   ['linspace', 1, 2, [2, None], _type], 2)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key', [[1, 2], None, _type], 2)

        with pytest.raises(TypeError):
            _numerical_param_value('good_key', ['linspace', 1, 2, None, _type], 2)


    @pytest.mark.parametrize('_type', allowed_types)
    @pytest.mark.parametrize('bad_points', (-1, 0))
    def test_rejects_integer_less_than_one(self, _type, bad_points):

        with pytest.raises(ValueError):
            _numerical_param_value('good_key', [[1,2], bad_points, _type], 2)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', ['linspace', 1, 2, bad_points, _type], 2)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key', [[1,2], [2, bad_points], _type], 2)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', ['linspace', 1, 2, [2, bad_points], _type], 2)

        with pytest.raises(ValueError):
            _numerical_param_value('good_key', [[1,2], [bad_points, 2], _type], 2)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', ['linspace', 1, 2, [bad_points, 2], _type], 2)


class TestPointsWhenFixed:


    @pytest.mark.parametrize('_type', ('fixed_integer', 'fixed_float'))
    @pytest.mark.parametrize('_points, total_passes',
         (
             (3, 1),
             (3, 3),
             ([3], 1),
             ([3,3,3], 3)
         )
    )
    def test_fixed_accepts_points_equals_len_grid(self,
        _type, _points, total_passes):

        out = _numerical_param_value(
            'good_key', [[1, 2, 3], _points, _type], total_passes
        )

        assert isinstance(out, list)
        assert out == [[1, 2, 3], [3 for _ in range(total_passes)], _type]


    @pytest.mark.parametrize('_type', ('fixed_integer', 'fixed_float'))
    @pytest.mark.parametrize('pass_num, _points, total_passes',
         (
             (1, 1, 1),
             (2, 1, 3),
             (3, [1], 1),
             (4, [3,1,1], 3),
             (5, [3,3,1], 3),
             (6, [3,1,3], 3),
             (7, [1,1,1], 3)
         )
    )
    def test_fixed_accepts_points_equals_1_after_first_pass(self,
        pass_num, _type, _points, total_passes):

        def _OUT(_points, _type, total_passes):
            return _numerical_param_value(
                'good_key', [[1, 2, 3], _points, _type], total_passes
            )

        if pass_num in [6]:
            with pytest.raises(ValueError):
                _OUT(_points, _type, total_passes)
        elif pass_num in [1, 2, 3, 4, 5, 7]:
            out = _OUT(_points, _type, total_passes)
            assert isinstance(out, list)
            if pass_num in [1, 3]:
                answer_points = [3]
            elif pass_num in [2, 4, 7]:
                answer_points = [3, 1, 1]
            elif pass_num == 5:
                answer_points = [3, 3, 1]

            assert out == [[1, 2, 3], answer_points, _type]


    @pytest.mark.parametrize('v1', (3, 4))
    @pytest.mark.parametrize('v2', (3, 4))
    @pytest.mark.parametrize('v3', (3, 4))
    def test_fixed_rejects_points_not_equal_len_grid_or_1(self, v1, v2, v3):

        # for points after first pass (first pass points is overwritten by
        # actual points in first grid)

        if v2 == 3 and v3 == 3:   # v1 can equal anything > 0
            # v1 will always be set to 3
            pytest.skip(reason=f"this combination will pass")

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[1 ,2, 3], 4, 'fixed_integer'], 3)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[1, 2, 3], [v1, v2, v3], 'fixed_integer'], 3)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[1.1, 2.1, 3.1], 4, 'fixed_float'], 3)

        with pytest.raises(ValueError):
            _numerical_param_value(
                'good_key', [[1.1, 2.1, 3.1], [v1, v2, v3], 'fixed_float'], 3)

class TestPointsAsInteger:

    def test_hard_soft_accepts_points_equals_len_grid(self):

        # accepts first points == len(grid) and any other points elsewhere

        _numerical_param_value(
            'good_key', [[11, 21, 31], 3, 'hard_integer'], 3)

        _numerical_param_value(
            'good_key', [[1.1, 2.1, 3.1], 3, 'hard_float'], 3)

        _numerical_param_value(
            'good_key', [[11 ,21, 13], 3, 'soft_integer'], 3)

        _numerical_param_value(
            'good_key', [[1.1, 2.1, 3.1], 3, 'soft_float'], 3)


    @pytest.mark.parametrize('v1', (2, 4, 5))
    def test_hard_soft_accepts_points_not_equal_len_grid(self, v1):

        # but rejects soft # points <= 2

        _numerical_param_value('good_key', [[11 ,21, 13], v1, 'hard_integer'], 3)

        _numerical_param_value('good_key', [[1.1, 2.1, 3.1], v1, 'hard_float'], 3)

        if v1 >= 3:
            _numerical_param_value('good_key', [[11 ,21, 13], v1, 'soft_integer'], 3)

            _numerical_param_value('good_key', [[1.1, 2.1, 3.1], v1, 'soft_float'], 3)

        elif v1 < 3:
            with pytest.raises(ValueError):
                _numerical_param_value('good_key',
                                       [[11, 21, 13], v1, 'soft_integer'], 3)

            with pytest.raises(ValueError):
                _numerical_param_value('good_key',
                                        [[1.1, 2.1, 3.1], v1, 'soft_float'], 3)

class TestPointsAsListType:

    @pytest.mark.parametrize('_type',
        ('hard_integer', 'hard_float', 'soft_integer', 'soft_float'))
    @pytest.mark.parametrize('v1', (2, 3, 4, 5))
    @pytest.mark.parametrize('v2', (2, 3, 4, 5))
    @pytest.mark.parametrize('v3', (2, 3, 4, 5))
    def test_hard_soft_conditionally_accepts_any_points(self,
        v1, v2, v3, _type):

        # soft rejects anywhere points == 2 but otherwise accepts any
        # of these values, _numerical_params always overwrites v1 with
        # actual points in first grid, and can accept any number in the
        # remaining positions

        if 'soft' in _type and (v1 == 2 or v2 == 2 or v3 == 2):
            with pytest.raises(ValueError):
                _numerical_param_value(
                    'good_key',
                    [[11, 12, 13], [v1, v2, v3], _type],
                    3
                )
        else:
            out = _numerical_param_value(
                'good_key', [[11, 12, 13], [v1, v2, v3], _type], 3)

            assert isinstance(out, list)
            assert out == [[11, 12, 13], [3, v2, v3], _type]


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










