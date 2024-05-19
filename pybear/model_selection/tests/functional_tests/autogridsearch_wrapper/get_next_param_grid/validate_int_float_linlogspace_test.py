# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _validate_int_float_linlogspace import _validate_int_float_linlogspace as vifl


@pytest.fixture
def float_module_names():
    return ['float_linspace', '_float_logspace']

@pytest.fixture
def int_module_names():
    return [
            '_int_linspace_unit_gap',
            '_int_logspace_unit_gap',
            '_int_linspace_gap_gt_1',
            '_int_logspace_gap_gt_1'
    ]



class TestValidateIntFloatLinLogspace:


    # module_name ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *

    @pytest.mark.parametrize('junk_name',
        (True, np.pi, [1, 2], (1, 2), {1, 2}, {'a': 1}, lambda x: x, min, None)
    )
    def test_rejects_junk_module_name(self, junk_name):
        with pytest.raises(TypeError):
            vifl([1,2,3], 2, False, False, 1, 3, 3, junk_name)

    @pytest.mark.parametrize('bad_name',
        ('garbage', 'more garbage', 'junk', 'and more junk')
    )
    def test_rejects_bad_module_name(self, bad_name):
        with pytest.raises(ValueError):
            vifl([1,2,3], 2, False, False, 1, 3, 3, bad_name)


    @pytest.mark.parametrize('grid, is_logspace, module',
        (
            ([10, 20, 30], False, '_float_linspace'),
            ([1, 10, 100,], 1.0, '_float_logspace'),
            ([1,2,3,4], False, '_int_linspace_unit_gap'),
            ([1,3,5,7], False, '_int_linspace_gap_gt_1'),
            ([1,10,100,1000], 1.0, '_int_logspace_unit_gap'),
            ([1, 100, 10000], 2.0, '_int_logspace_gap_gt_1')
         )
    )
    def test_accepts_good_module_name(self, grid, is_logspace, module):
        vifl(grid, 2, is_logspace, False, min(grid), max(grid), 3, module)

    # END module_name ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # search points ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('bad_points',
        (1, True, np.pi, {'a': 1}, lambda x: x, min, 'junk', None)
    )
    def test_rejects_non_list_like_grid(self, bad_points, float_module_names):
        with pytest.raises(TypeError):
            vifl(bad_points, 1, False, False, 1, 10, 3, float_module_names[0])


    @pytest.mark.parametrize('good_points',
        ([2, 3, 4], (2, 3, 4), {2, 3, 4}, np.array([2,3,4]))
    )
    def test_accepts_list_like_grid(self, good_points, int_module_names):
        vifl(good_points, 1, False, False, 1, 10, 3, int_module_names[0])


    @pytest.mark.parametrize('bad_values',
        (list('abc'), [[1], [2], [3]], [None, None, None])
    )
    def test_rejects_non_num_in_grid(self, bad_values, int_module_names):
        with pytest.raises(TypeError):
            vifl(bad_values, 1, False, False, 1, 10, 3, int_module_names[1])


    def test_int_rejects_floats_in_grid(self, int_module_names):
        with pytest.raises(TypeError):
            vifl([1.1, 1.2, 1.3], 1, False, False, 1, 10, 3, int_module_names[2])


    def test_int_accepts_int_in_grid(self, int_module_names):
        vifl([1, 2, 3], 1, False, False, 1, 10, 3, int_module_names[0])


    def test_float_accepts_floats_in_grid(self, float_module_names):
        vifl([1.1, 1.2, 1.3], 1, False, False, 1, 10, 3, float_module_names[0])


    def test_float_accepts_int_in_grid(self, float_module_names):
        vifl([1, 2, 3], 1, False, False, 1, 10, 3, float_module_names[0])


    def test_rejects_lt_universal_bound_in_grid(self, float_module_names,
                                                int_module_names):
        with pytest.raises(ValueError):
            vifl([-1, 2, 3], 1, False, False, 1, 3, 0, float_module_names[1])

        with pytest.raises(ValueError):
            vifl([0, 2, 3], 1, False, False, 1, 3, 0, int_module_names[3])


    def test_rejects_duplicates_in_grid(self):
        with pytest.raises(ValueError):
            vifl([10, 20, 30, 30], 0, False, False, 10, 40, 0, '_float_linspace')


    def test_rejects_len_grid_lt_3(self):
        with pytest.raises(ValueError):
            vifl([10, 20], 0, False, False, 10, 40, 0, '_float_linspace')

    # END search points ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # _posn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('bad_posn',
        (True, np.pi, [1,2], (1,2), {1,2}, {'a': 1}, lambda x: x, min, 'junk', None)
    )
    def test_rejects_junk_posn(self, bad_posn):
        with pytest.raises(TypeError):
            vifl([1, 2, 3], bad_posn, False, False, 1, 3, 3, '_int_linspace_unit_gap')

    def test_rejects_float_posn(self):
        with pytest.raises(TypeError):
            vifl([1, 2, 3], 3.14, False, False, 1, 3, 3, '_int_linspace_unit_gap')

    def test_rejects_negative_posn(self):
        with pytest.raises(ValueError):
            vifl([1, 2, 3], -1, False, False, 1, 3, 3, '_int_linspace_unit_gap')

    def test_rejects_out_of_bounds(self):
        with pytest.raises(ValueError):
            vifl([1, 2, 3], 9, False, False, 1, 3, 3, '_int_linspace_unit_gap')

    # END _posn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # _is_logspace ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('bad_is_logspace',
        ([1,2], (1,2), {1,2}, {'a': 1}, lambda x: x, min, 'junk', None)
    )
    def test_is_logspace_rejects_non_num_non_bool(self, bad_is_logspace):
        with pytest.raises(TypeError):
            vifl([1, 10, 100], 0, bad_is_logspace, False, 1, 100, 3,
                 '_float_logspace')

    @pytest.mark.parametrize('bad', (0, -1))
    def test_is_logspace_rejects_lt_1(self, bad):
        with pytest.raises(TypeError):
            vifl([1, 10, 100], 0, bad, False, 1, 100, 3, '_float_logspace')


    @pytest.mark.parametrize('is_logspace, grid, module',
        (
        (False, [1, 2, 3, 4], '_int_linspace_unit_gap'),
        # (1, [1, 10, 100, 1000], '_int_logspace_unit_gap'),
        (1.0, [1, 10, 100, 1000], '_int_logspace_unit_gap')
        )
    )
    def test_accepts_false_and_positive_numbers_1(self, is_logspace, grid, module):
        vifl(grid, 1, is_logspace, False, grid[0], grid[-1], 3, module)


    @pytest.mark.parametrize('is_logspace, grid, module',
        (
        (False, [10, 20, 30, 40], '_int_linspace_gap_gt_1'),
        # (2, [1, 100, 10000], '_int_logspace_gap_gt_1'),
        (2.0, [1, 100, 10000], '_int_logspace_gap_gt_1')
        )
    )
    def test_accepts_false_and_positive_numbers_2(self, is_logspace, grid, module):
        vifl(grid, 1, is_logspace, False, grid[0], grid[-1], 11, module)


    def test_rejects_logspace_misdiagnosis_1(self):
        with pytest.raises(ValueError):
            vifl([1, 2, 3, 4], 1, 1.0, False, 1, 4, 3,
                 '_int_linspace_unit_gap')


    def test_rejects_logspace_misdiagnosis_2(self):
        with pytest.raises(ValueError):
            vifl([1, 10, 100], 1, False, False, 1, 100, 3, '_int_linspace_unit_gap')

    # END _is_logspace ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # _is_hard ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('bad_is_hard',
        (0, np.pi, [1, 2], (1, 2), {1, 2}, {'a': 1}, min, lambda x: x, None, 'junk')
    )
    def test_rejects_non_bool_is_hard(self, bad_is_hard):
        with pytest.raises(TypeError):
            vifl([2, 3, 4, 5], 1, False, bad_is_hard, 2, 5, 3,
                '_int_linspace_unit_gap')

    @pytest.mark.parametrize('good', (True, False))
    def test_accepts_bool_is_hard(self, good):
        vifl([2, 3, 4, 5], 1, False, good, 2, 5, 3, '_int_linspace_unit_gap')

    # END _is_hard ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




    # _hard_min, _hard_max ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('bad_value',
        ([1, 2], (1, 2), {1, 2}, {'a': 1}, min, lambda x: x, None, 'junk')
    )
    def test_int_rejects_junk_hard_min_hard_max(self, bad_value):
        _grid = [2, 3, 4]
        _module = '_int_linspace_unit_gap'

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, False, False, bad_value, 4, 3, _module)

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, False, False, 2, bad_value, 3, _module)


        _grid = np.power(10, [1, 3, 5])
        _module = '_int_logspace_gap_gt_1'

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, 2.0, False, bad_value, 4, 3, _module)

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, 2.0, False, 2, bad_value, 3, _module)


    @pytest.mark.parametrize('bad_value',
        (np.pi, 10**2.34)
    )
    def test_int_rejects_non_int_hard_min_hard_max(self, bad_value):
        _grid = [2, 3, 4]
        _module = '_int_linspace_unit_gap'

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, False, False, bad_value, 4, 3, _module)

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, False, False, 2, bad_value, 3, _module)


        _grid = np.power(10, [1, 3, 5])
        _module = '_int_logspace_gap_gt_1'

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, 1.0, False, bad_value, 4, 3, _module)

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, 1.0, False, 2, bad_value, 3, _module)


    @pytest.mark.parametrize('bad', (0, -1))
    def test_rejects_hard_min_hard_max_lt_universal_lower(self, bad):
        _grid = [2, 3, 4]

        _module = '_int_linspace_unit_gap'
        with pytest.raises(ValueError):
            _grid_out = vifl(_grid, 2, False, False, bad ,4, 3, _module)

        with pytest.raises(ValueError):
            _grid_out = vifl(_grid, 2, False, False, 2, bad, 3, _module)

        _module = '_float_linspace'
        if bad != 0:
            with pytest.raises(ValueError):
                _grid_out = vifl(_grid, 2, False, False, bad ,4, 3, _module)

            with pytest.raises(ValueError):
                _grid_out = vifl(_grid, 2, False, False, 2, bad, 3, _module)


    @pytest.mark.parametrize('bad', (3, 5))
    def test_rejects_grid_lt_hard_min(self, bad):
        _grid = [2, 4, 6]

        _module = '_int_linspace_unit_gap'
        with pytest.raises(ValueError):
            _grid_out = vifl(_grid, 2, False, False, bad ,4, 3, _module)

        _module = '_float_linspace'
        if bad != 0:
            with pytest.raises(ValueError):
                _grid_out = vifl(_grid, 2, False, False, bad ,4, 3, _module)


    @pytest.mark.parametrize('bad', (3, 5))
    def test_rejects_grid_gt_hard_max(self, bad):
        _grid = [2, 4, 6]

        _module = '_int_linspace_unit_gap'
        with pytest.raises(ValueError):
            _grid_out = vifl(_grid, 2, False, False, 2, bad, 3, _module)

        _module = '_float_linspace'
        if bad != 0:
            with pytest.raises(ValueError):
                _grid_out = vifl(_grid, 2, False, False, 2, bad, 3, _module)
    # END _hard_min, _hard_max ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # _points ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('bad_value',
        (np.pi, [1, 2], (1, 2), {1, 2}, {'a': 1}, min, lambda x: x, None, 'junk')
    )
    def test_rejects_non_int_points(self, bad_value):
        _grid = [2, 3, 4]
        _module = '_int_linspace_unit_gap'

        with pytest.raises(TypeError):
            _grid_out = vifl(_grid, 2, False, False, 2, 4, bad_value, _module)


    @pytest.mark.parametrize('bad', (-1, 0, 1, 2))
    def test_rejects_points_lt_3(self, bad):
        _grid = [2, 3, 4]
        _module = '_int_linspace_unit_gap'

        with pytest.raises(ValueError):
            _grid_out = vifl(_grid, 2, False, False, 2, 4, bad, _module)


    # END _hard_min, _hard_max, _points ** * ** * ** * ** * ** * ** * ** * ** *






















