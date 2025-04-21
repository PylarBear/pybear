# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _params_numerical import _val_numerical_param_value



allowed_types = [
    'fixed_integer', 'fixed_float', 'hard_integer',
    'hard_float', 'soft_integer', 'soft_float'
]



@pytest.mark.parametrize('total_passes', (1,3))
class TestNumericalParamKey:


    def test_accepts_str(self, total_passes):

        value = [[1, 2, 3, 4], [4, 4, 4], 'hard_integer']

        value[-2] = [value[-2][0] for _ in range(total_passes)]
        assert _val_numerical_param_value(
            'some_string', value, total_passes
        ) is None


@pytest.mark.parametrize('total_passes', (1, 3))
class TestNumericalParamValueOuterContainer:


    @pytest.mark.parametrize('list_like', ('list', 'tuple', 'np.array'))
    def test_accepts_list_like(self, list_like, total_passes):

        value = [[1,2,3], [3 for _ in range(total_passes)], 'soft_float']

        if list_like == 'list':
            list_like = list(value)
        elif list_like == 'tuple':
            list_like = tuple(value)
        elif list_like == 'np.array':
            list_like = np.array(value, dtype=object)
        else:
            raise Exception

        assert _val_numerical_param_value(
            'good_key', list_like, total_passes
        ) is None


@pytest.mark.parametrize('_type', allowed_types)
@pytest.mark.parametrize('total_passes', (1, 3))
class TestGridAsListOfValues:


    @pytest.mark.parametrize('non_num',
        (min, 'junk', lambda x: x, {'a': 1}, [1,2], (1,2), {1,2})
    )
    def test_rejects_non_numeric_inside(self, non_num, total_passes, _type):

        with pytest.raises(TypeError):
            _val_numerical_param_value(
                'good_key',
                [[non_num, 2, 3], [3 for _ in range(total_passes)], _type],
                total_passes
            )


    @pytest.mark.parametrize('_points', (4, 6))
    def test_rejects_non_int_log_gaps(self, total_passes, _type, _points):

        # also implies log gaps less than 1 are rejected

        _params = {
            'a': [
                np.logspace(-4, 4, _points),
                [_points for _ in range(total_passes)],
                _type
            ]
        }

        with pytest.raises(ValueError):
            _val_numerical_param_value('a', _params['a'], 4)


@pytest.mark.parametrize('total_passes', (1, 3))
@pytest.mark.parametrize('_type', ('soft_integer', 'hard_integer', 'fixed_integer'))
class TestIntGridAsListOfValues:


    @pytest.mark.parametrize('_grid', ([1,2,np.pi], [1e-6, 1e-6, 1e-4]))
    def test_integer_dtype_rejects_float(self, total_passes, _type, _grid):

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key',
                [_grid, [3 for _ in range(total_passes)], _type],
                total_passes
            )


    def test_int_rejects_lt_one_when_not_fixed(self, total_passes, _type):

        if 'fixed' in _type:
            assert _val_numerical_param_value(
                'good_key',
                [[0,1,2], [3 for _ in range(total_passes)], _type],
                total_passes
            ) is None
        else:
            with pytest.raises(ValueError):
                _val_numerical_param_value(
                    'good_key',
                    [[0,1,2], [3 for _ in range(total_passes)], _type],
                    total_passes
                )


    def test_int_dtype_rejects_bool(self, total_passes, _type):

        with pytest.raises(TypeError):
            _val_numerical_param_value(
                'good_key',
                [[True, False], [2 for _ in range(total_passes)], _type],
                total_passes
            )


@pytest.mark.parametrize('_type', ('soft_float', 'hard_float', 'fixed_float'))
@pytest.mark.parametrize('total_passes', (1, 3))
class TestFloatGridAsListOfValues:


    def test_float_rejects_lt_zero_when_not_fixed(self, total_passes, _type):

        if 'fixed' in _type:
            assert _val_numerical_param_value(
                'good_key',
                [[-1, 0, 1], [3 for _ in range(total_passes)], _type],
                total_passes
            ) is None
        else:
            with pytest.raises(ValueError):
                _val_numerical_param_value(
                    'good_key',
                    [[-1, 0, 1], [3 for _ in range(total_passes)], _type],
                    total_passes
                )


    @pytest.mark.parametrize('value', (0, 1, np.pi))
    def test_float_dtype_accepts_any_other_number(
            self, value, total_passes, _type
    ):

        points =[3 for _ in range(total_passes)]

        assert _val_numerical_param_value(
            'good_key',
            [[1,2,value], points, _type],
            total_passes
        ) is None


    def test_float_dtype_rejects_bool(self, total_passes, _type):

        with pytest.raises(TypeError):
            _val_numerical_param_value(
                'good_key',
                [[True, False], [2 for _ in range(total_passes)], _type],
                total_passes
            )


@pytest.mark.parametrize('_type', allowed_types)
@pytest.mark.parametrize('total_passes', (1, 3))
class TestPoints:


    @pytest.mark.parametrize('non_list_type',
        (np.pi, True, min, 'junk', lambda x: x, {'a': 1})
    )
    def test_rejects_non_int_non_list_type(
        self, total_passes, _type, non_list_type
    ):

        with pytest.raises(TypeError):
            _val_numerical_param_value(
                'good_key', [[1,2,3], non_list_type, _type], total_passes
            )


    @pytest.mark.parametrize('list_type', (list, tuple, np.array))
    def test_accepts_list_type(self, total_passes, list_type, _type):

        assert _val_numerical_param_value(
            'good_key',
            [[1,2,3], list_type([3 for _ in range(total_passes)]), _type],
            total_passes
        ) is None


    def test_accepts_integer_gte_one(self, total_passes, _type):

        assert _val_numerical_param_value(
            'good_key',
            [[1, 2, 3], 3, _type],
            total_passes
        ) is None


    def test_rejects_none(self, total_passes, _type):

        with pytest.raises(TypeError):
            _val_numerical_param_value(
                'good_key',
                [[1, 2], [None, 2, 2][:total_passes], _type],
                total_passes
            )


    @pytest.mark.parametrize('bad_points', (-1, 0))
    def test_rejects_integer_less_than_one(
        self, _type, total_passes, bad_points
    ):

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key', [[1,2], bad_points, _type], total_passes
            )

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key', [[1,2], [2, bad_points], _type], total_passes
            )

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key', [[1,2], [bad_points, 2], _type], total_passes
            )


@pytest.mark.parametrize('_type', ('fixed_integer', 'fixed_float'))
class TestPointsWhenFixed:


    @pytest.mark.parametrize('_points, total_passes',
         (
             (3, 1),
             (3, 3),
             ([3], 1),
             ([3,3,3], 3)
         )
    )
    def test_fixed_accepts_points_equals_len_grid(
        self, _type, _points, total_passes
    ):

        assert _val_numerical_param_value(
            'good_key', [[1, 2, 3], _points, _type], total_passes
        ) is None


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
    def test_fixed_accepts_points_equals_1_after_first_pass(
        self, pass_num, _type, _points, total_passes
    ):

        if pass_num in [6]:
            with pytest.raises(ValueError):
                _val_numerical_param_value(
                    'good_key', [[1, 2, 3], _points, _type], total_passes
                )
        elif pass_num in [1, 2, 3, 4, 5, 7]:
            assert _val_numerical_param_value(
                'good_key', [[1, 2, 3], _points, _type], total_passes
            ) is None


    @pytest.mark.parametrize('v1', (3, 4))
    @pytest.mark.parametrize('v2', (3, 4))
    @pytest.mark.parametrize('v3', (3, 4))
    def test_fixed_rejects_points_not_equal_len_grid_or_1(
        self, _type, v1, v2, v3
    ):

        # for points after first pass (first pass points is overwritten by
        # actual points in first grid)

        if v2 == 3 and v3 == 3:   # v1 can equal anything > 0
            # v1 will always be set to 3
            pytest.skip(reason=f"this combination will pass")

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key', [[1 ,2, 3], 4, _type], 3)

        with pytest.raises(ValueError):
            _val_numerical_param_value(
                'good_key', [[1, 2, 3], [v1, v2, v3], _type], 3)


@pytest.mark.parametrize('_type',
    ('hard_integer', 'hard_float', 'soft_integer', 'soft_float')
)
class TestPointsAsInteger:


    def test_hard_soft_accepts_points_equals_len_grid(self, _type):

        # accepts first points == len(grid) and any other points elsewhere

        assert _val_numerical_param_value(
            'good_key', [[11, 21, 31], 3, _type], 3
        ) is None


    @pytest.mark.parametrize('v1', (2, 4, 5))
    def test_hard_soft_accepts_points_not_equal_len_grid(self, _type, v1):

        # but rejects soft # points <= 2

        if 'hard' in _type:
            assert _val_numerical_param_value(
                'good_key', [[11 ,21, 13], v1, _type], 3
            ) is None
        elif 'soft' in _type:
            if v1 >= 3:
                assert _val_numerical_param_value(
                    'good_key', [[11 ,21, 13], v1, _type], 3
                ) is None

            elif v1 < 3:
                with pytest.raises(ValueError):
                    _val_numerical_param_value(
                        'good_key', [[11, 21, 13], v1, _type], 3
                    )
        else:
            raise Exception


class TestPointsAsListType:

    @pytest.mark.parametrize('_type',
        ('hard_integer', 'hard_float', 'soft_integer', 'soft_float'))
    @pytest.mark.parametrize('v1', (2, 3, 4, 5))
    @pytest.mark.parametrize('v2', (2, 3, 4, 5))
    @pytest.mark.parametrize('v3', (2, 3, 4, 5))
    def test_hard_soft_conditionally_accepts_any_points(
        self, v1, v2, v3, _type
    ):

        # soft rejects anywhere points == 2 but otherwise accepts any
        # of these values, _numerical_params always overwrites v1 with
        # actual points in first grid, and can accept any number in the
        # remaining positions

        if 'soft' in _type and (v1 == 2 or v2 == 2 or v3 == 2):
            with pytest.raises(ValueError):
                _val_numerical_param_value(
                    'good_key',
                    [[11, 12, 13], [v1, v2, v3], _type],
                    3
                )
        else:
            assert _val_numerical_param_value(
                'good_key', [[11, 12, 13], [v1, v2, v3], _type], 3
            ) is None








