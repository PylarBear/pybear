# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from copy import deepcopy
import numpy as np

from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _shift._shift_points_and_passes import _shift_points_and_passes





class TestShiftPoints:

    # no validation


    # since handling is different for 'string' and ('soft_float',
    # 'hard_float', 'fixed_float', 'soft_integer', 'hard_integer',
    # 'fixed_integer'), and the ones in parentheses are all handled the
    # same, just test 'string' and one of the others.

    @pytest.mark.parametrize('total_passes', (2, 3, 4))
    @pytest.mark.parametrize('number_of_params', (1, 3, 10))
    @pytest.mark.parametrize('total_passes_is_hard', (True, False))
    def test_accuracy(self, total_passes, number_of_params, total_passes_is_hard):

        # build params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        _params = {}

        _keys = list('abcdefghijklmn'[:number_of_params])

        for _key in _keys:

            _random_dtype = np.random.choice(
                            ['string', 'fixed_integer', 'soft_float'],
                            size=1
            )[0]
            _random_grid_size = np.random.randint(1,10)

            if _random_dtype == 'string':
                _grid = list('abcdefghijklmn'[:_random_grid_size])
                _shrink_pass = np.random.randint(1,10)
                _params[_key] = [_grid, _shrink_pass, _random_dtype]
            else:
                _grid = np.arange(1, _random_grid_size+1).tolist()
                _points = np.random.randint(2, 11, total_passes).tolist()
                _params[_key] = [_grid, _points, _random_dtype]


        del _random_dtype, _random_grid_size, _grid

        # END build params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        _pass = np.random.randint(1, total_passes)   # cannot be pass 0


        # ** * ** *

        out_params = _shift_points_and_passes(
                                                deepcopy(_params),
                                                _pass,
                                                total_passes_is_hard
        )

        # ** * ** *


        # build expected_params ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        expected_params = _params

        for _param in expected_params:
            if expected_params[_param][-1] == 'string':
                expected_params[_param][-2] += 1
            else:
                expected_params[_param][-2].insert(
                                _pass, expected_params[_param][-2][_pass-1]
                )

                if total_passes_is_hard:
                    expected_params[_param][-2] = expected_params[_param][-2][:-1]

        # END build expected_params ** * ** * ** * ** * ** * ** * ** * ** * **


        assert out_params == expected_params






















