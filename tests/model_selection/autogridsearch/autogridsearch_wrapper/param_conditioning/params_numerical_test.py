# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._params_numerical import _cond_numerical_param_value

import pytest



class TestCondNumericalParamValueAccuracy:

    # there is no validation going into this module

    # out container and grid container must be lists already
    # last slot must be lower-case already


    @pytest.mark.parametrize('outer_container', (list, ))
    @pytest.mark.parametrize('base_grid',
        (np.logspace(-5, 5, 11), np.linspace(100, 1000, 11), [1,2,3])
    )
    @pytest.mark.parametrize('grid_container', (list, ))
    @pytest.mark.parametrize('base_points, total_passes',
        ([[3,1,1], 3], [4, 3], [[3,3,3], 3])
    )
    @pytest.mark.parametrize('points_container', (list, tuple, np.ndarray))
    @pytest.mark.parametrize('paramtype',
        ('soft_float', 'hard_float', 'fixed_float', 'soft_integer',
         'hard_integer', 'fixed_integer')
    )
    def test_accepts_list_like(
        self, outer_container, base_grid, grid_container, base_points,
        total_passes, points_container, paramtype
    ):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # skip impossible conditions
        # in this particular case, we have integer with logspace < 1,
        # which would be blocked by validation
        if min(base_grid) < 1 and 'integer' in paramtype:
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


        out = _cond_numerical_param_value(
            _param_value,
            _total_passes=total_passes
        )

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(out, list)
        assert len(out) == 3
        # * * * * * * * * * * * * * *
        assert isinstance(out[0], list)
        if grid_container is set:
            assert np.array_equal(out[0], sorted(list(set(base_grid))))
        else:
            assert np.array_equal(out[0], base_grid)
        # * * * * * * * * * * * * * *
        assert isinstance(out[1], list)
        if isinstance(base_points, numbers.Number):
            _ref_base_points = [base_points for i in range(total_passes)]
        else:
            _ref_base_points = base_points.copy()
        _ref_base_points[0] = len(base_grid)
        assert np.array_equal(out[1], _ref_base_points)
        # * * * * * * * * * * * * * *
        assert out[2] == paramtype









