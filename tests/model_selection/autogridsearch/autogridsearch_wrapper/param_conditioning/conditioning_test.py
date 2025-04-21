# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._conditioning import _conditioning

import numbers

import numpy as np

import pytest



class TestConditioning:

    # this module is a hub for the conditioning submodules, whose
    # accuracy are tested elsewhere. just do basic checks to make sure
    # this module works.

    @pytest.mark.parametrize('_total_passes', (2, 3))
    @pytest.mark.parametrize('_max_shifts', (None, 3))
    @pytest.mark.parametrize('_inf_shrink_pass', (123_456, 1_000_000_000))
    @pytest.mark.parametrize('_inf_max_shifts', (985682, 1_000_000))
    def test_accuracy(
        self, _total_passes, _max_shifts, _inf_shrink_pass, _inf_max_shifts
    ):

        _params = {
            'param_a': [{'a', 'b', 'c'}, None, 'StRiNg'],
            'param_b': ({True, False}, 2, 'BOOL'),
            'param_c': [(1,2,3,4), [4,4,1,1], 'fixed_integer'],
            'param_d': (np.logspace(-5, 5, 11), 11, 'SOFT_float')
        }

        out_params, out_total_passes, out_max_shifts = _conditioning(
            _params,
            _total_passes,
            _max_shifts,
            _inf_shrink_pass,
            _inf_max_shifts
        )


        key_key = {0: 'param_a', 1:'param_b', 2:'param_c', 3:'param_d'}
        type_key = {0: 'string', 1: 'bool', 2: 'fixed_integer', 3: 'soft_float'}


        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # out_params
        assert isinstance(out_params, dict)
        for idx, (key, value) in enumerate(out_params.items()):
            # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
            # params keys
            assert isinstance(key, str)
            assert key == key_key[idx]
            # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
            # params values
            assert isinstance(value, list)
            # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
            # values idx 0
            assert isinstance(value[0], list)
            assert np.array_equal(value[0], list(_params[key][0]))
            # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
            # values idx 1
            if value[-1] in ['string', 'bool']:
                assert isinstance(value[1], numbers.Integral)
                if _params[key][1] is None:
                    assert value[1] == _inf_shrink_pass
            else:   # numeric params
                assert isinstance(value[1], list)
                assert all(map(isinstance, value[1], (int for _ in value[1])))
                assert len(value[1]) == out_total_passes
            # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
            # values idx 2
            assert isinstance(value[-1], str)
            assert value[-1] == type_key[idx]
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # out_total_passes
        assert isinstance(out_total_passes, numbers.Integral)
        # because number of passes implied by param_c points list....
        assert out_total_passes == len(out_params['param_c'][1])
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # out_max_shifts
        assert isinstance(out_max_shifts, numbers.Integral)
        assert out_max_shifts == (_max_shifts or _inf_max_shifts)
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




