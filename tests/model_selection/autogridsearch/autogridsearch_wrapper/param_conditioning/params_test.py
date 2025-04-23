# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._params import _cond_params

import pytest



class TestCondParams:


    @staticmethod
    @pytest.fixture
    def good_dict_params():
        return {
            'param_a': [['a', 'b', 'c'], 3, 'fixed_string'],
            'param_b': [np.logspace(1, 3, 3), [3, 11, 6], 'soft_float'],
            'param_c': [[True, False], 2, 'fixed_bool']
        }


    @staticmethod
    @pytest.fixture
    def answer_good_dict_params():
        return {
            'param_a': [['a', 'b', 'c'], [3, 3, 3], 'fixed_string'],
            'param_b': [[10.0, 100.0, 1000.0], [3, 11, 6], 'soft_float'],
            'param_c': [[True, False], [2, 2, 2], 'fixed_bool']
        }

    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('outer_container', (list, tuple, np.ndarray))
    @pytest.mark.parametrize('base_grid',
        (np.logspace(-5, 5, 11), np.linspace(100, 1000, 11), [1,2,3])
    )
    @pytest.mark.parametrize('grid_container', (list, tuple, set, np.ndarray))
    @pytest.mark.parametrize('base_points, total_passes',
        ([[3,1,1], 3], [4, 3], [[3,3,3], 3])
    )
    @pytest.mark.parametrize('points_container', (list, tuple, np.ndarray))
    @pytest.mark.parametrize('paramtype',
        ('SOFT_FLOAT', 'hard_FLOAT', 'fixed_float', 'SoFt_InTeGer',
         'hard_integer', 'FIXED_integer')
    )
    def test_accepts_list_like(
        self, outer_container, base_grid, grid_container, base_points,
        total_passes, points_container, paramtype
    ):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # skip impossible conditions
        # in this particular case, we have integer with logspace < 1,
        # which would be blocked by validation
        if min(base_grid) < 1 and 'integer' in paramtype.lower():
            pytest.skip(reason=f"impossible condition")
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _base_outer = [None, None, None]

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if grid_container in [list, tuple, set]:
            _base_outer[0] = grid_container(base_grid)
        elif grid_container is np.ndarray:
            _base_outer[0] = np.array(list(base_grid))
        else:
            raise Exception

        assert isinstance(_base_outer[0], grid_container)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if isinstance(base_points, numbers.Real):
            _base_outer[1] = base_points
        elif points_container in [list, tuple, set]:
            _base_outer[1] = points_container(base_points)
        elif points_container is np.ndarray:
            _base_outer[1] = np.array(list(base_points))
        else:
            raise Exception

        assert isinstance(_base_outer[0], grid_container)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _base_outer[2] = paramtype

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if outer_container in [list, tuple]:
            _param_value = outer_container(_base_outer)
        elif outer_container is np.ndarray:
            _param_value = np.array(list(_base_outer), dtype=object)
        else:
            raise Exception

        assert isinstance(_param_value, outer_container)


        out_params = _cond_params(
            {'good_param': _param_value},
            _total_passes=total_passes
        )

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(out_params, dict)
        assert len(out_params['good_param']) == 3
        # * * * * * * * * * * * * * *
        assert isinstance(out_params['good_param'][0], list)
        if grid_container is set:
            assert np.array_equal(
                out_params['good_param'][0],
                sorted(list(set(base_grid)))
            )
        else:
            assert np.array_equal(out_params['good_param'][0], base_grid)
        # * * * * * * * * * * * * * *
        assert isinstance(out_params['good_param'][1], list)
        if isinstance(base_points, numbers.Number):
            _ref_base_points = [base_points for i in range(total_passes)]
        else:
            _ref_base_points = base_points.copy()
        _ref_base_points[0] = len(base_grid)
        assert np.array_equal(out_params['good_param'][1], _ref_base_points)
        # * * * * * * * * * * * * * *
        assert out_params['good_param'][2] == paramtype.lower()


    # points len == passes from kwarg returns the same
    def test_same_returns_same(
        self, good_dict_params, answer_good_dict_params
    ):

        _params_out = _cond_params(good_dict_params, 3)

        assert _params_out == answer_good_dict_params



    # when str param only, kwarg total_passes is always returned
    @pytest.mark.parametrize('kwarg_passes', (1,2,3,4,5))
    def test_str_bool_params_only(self, kwarg_passes):

        _params = {
            'a': [['aa', 'bb', 'cc'], 2, 'fixed_string'],
            'b': [['dd', 'ee', 'ff'], 3, 'fixed_string'],
            'c': [['gg', 'hh', 'ii'], 4, 'fixed_string'],
            'd': [[True, False], 4, 'fixed_bool']
        }

        _params_out = _cond_params(_params, kwarg_passes)

        assert _params_out == _params


    @pytest.mark.parametrize('kwarg_passes', (1,2,3,4,5))
    def test_propagation_of_total_passes(self, kwarg_passes):

        _params = {
            'a': [np.logspace(-4, 4, 5), 5, 'soft_float'],
            'b': [np.linspace(100, 500, 5), 5, 'soft_integer'],
            'c': [[2, 3, 4, 5], 4, 'fixed_integer'],
            'd': [[True, False], 2, 'fixed_bool']
        }

        answer_params = {
            'a': [[1e-4, 1e-2, 1, 1e2, 1e4], [5] * kwarg_passes, 'soft_float'],
            'b': [[100, 200, 300, 400, 500], [5] * kwarg_passes, 'soft_integer'],
            'c': [[2, 3, 4, 5], [4] * kwarg_passes, 'fixed_integer'],
            'd': [[True, False], [2] * kwarg_passes, 'fixed_bool']
        }

        _params_out = _cond_params(_params, kwarg_passes)

        assert _params_out == answer_params











