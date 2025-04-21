# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._params_string import _cond_string_param_value

import pytest



class TestCondStringParamValueAccuracy:

    # there is no validation going into this module



    @pytest.mark.parametrize('outer_container', (list, tuple, np.ndarray))
    @pytest.mark.parametrize('base_grid', (list('abcde'), [None, 'qr', 'st']))
    @pytest.mark.parametrize('grid_container', (list, tuple, set, np.ndarray))
    @pytest.mark.parametrize('shrink_pass,inf_value',
        (
            (3, None),
            (3, 987_000),
            (None, 123_456),
            (None, 576_789)
        )
    )
    @pytest.mark.parametrize('paramtype', ('string', 'STRING'))
    def test_accepts_list_like(
        self, outer_container, base_grid, grid_container, shrink_pass, inf_value,
        paramtype
    ):

        _base_outer = [None, None, None]

        if grid_container in [list, tuple, set]:
            _base_outer[0] = grid_container(base_grid)
        elif grid_container is np.ndarray:
            _base_outer[0] = np.array(list(base_grid))
        else:
            raise Exception

        assert isinstance(_base_outer[0], grid_container)

        _base_outer[1] = shrink_pass

        _base_outer[2] = paramtype

        if outer_container in [list, tuple]:
            _param_value = outer_container(_base_outer)
        elif outer_container is np.ndarray:
            _param_value = np.array(list(_base_outer), dtype=object)
        else:
            raise Exception

        assert isinstance(_param_value, outer_container)


        out = _cond_string_param_value(_param_value, inf_value)

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(out, list)
        assert len(out) == 3
        assert isinstance(out[0], list)
        if grid_container is set:
            assert np.array_equal(out[0], list(set(base_grid)))
        else:
            assert np.array_equal(out[0], base_grid)
        assert isinstance(out[1], numbers.Integral)
        assert out[1] == (shrink_pass or inf_value)
        assert out[2] == paramtype.lower()









