# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza dont forget about set_params() blocks (max_recursions, ic, hab)

from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer as MCT

import numpy as np

import pytest



@pytest.fixture(scope='function')
def _args(_rows):
    return [_rows // 20]


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'ignore_float_columns': True,
        'ignore_non_binary_integer_columns': True,
        'ignore_columns': None,
        'ignore_nan': True,
        'delete_axis_0': True,
        'handle_as_bool': None,
        'reject_unseen_values': False,
        'max_recursions': 1,
        'n_jobs': -1
    }


@pytest.fixture(scope='function')
def _rows():
    return 200


@pytest.fixture(scope='function')
def X(_rows):
    return np.random.randint(0, 10, (_rows, 10))


@pytest.fixture(scope='function')
def y(_rows):
    return np.random.randint(0, 2, (_rows, 2), dtype=np.uint8)




class TestSetParams:

        # DEFAULT ARGS/KWARGS
        # _args = [_rows // 20]
        # _kwargs = {
        #     'ignore_float_columns': True,
        #     'ignore_non_binary_integer_columns': True,
        #     'ignore_columns': None,
        #     'ignore_nan': True,
        #     'delete_axis_0': True,
        #     'handle_as_bool': None,
        #     'reject_unseen_values': False,
        #     'max_recursions': 1,
        #     'n_jobs': -1
        # }


    def test_excepts_for_unknown_param(self, _args, _kwargs):

        TestCls = MCT(*_args, **_kwargs)

        with pytest.raises(ValueError):
            TestCls.set_params(garbage=1)


    def test_set_params_correctly_applies(self, X, y, _args, _kwargs):

        _first_params = {'count_threshold': _args[0]} | _kwargs

        TestCls = MCT(*_args, **_kwargs)

        # assert TestCls initiated correctly
        for _param, _value in _first_params.items():
            assert getattr(TestCls, _param) == _value

        # set new params before fit
        _scd_params = {'count_threshold': _args[0]} | _kwargs
        _scd_params['count_threshold'] = 39
        _scd_params['ignore_float_columns'] = False
        _scd_params['ignore_non_binary_integer_columns'] = False
        _scd_params['ignore_columns'] = [0, 1]
        _scd_params['ignore_nan'] = False
        _scd_params['delete_axis_0'] = False
        _scd_params['handle_as_bool'] = [2, 3]
        _scd_params['reject_unseen_values'] = True
        _scd_params['max_recursions'] = 1   # <==== no change
        _scd_params['n_jobs'] = 2

        TestCls.set_params(**_scd_params)

        # assert new values set correctly
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


        # do a fit(), then assert all of the hidden params are set correctly
        TestCls.fit(X, y)

        for _param, _value in _scd_params.items():
            try:
                if hasattr(TestCls, _param):
                    assert getattr(TestCls, _param) == _value
                else:
                    assert getattr(TestCls, f"_" + _param) == _value
            except:
                assert np.array_equal(getattr(TestCls, f"_" + _param), _value)


    def test_equality_set_params_before_and_after_fit(
        self, X, y, _args, _kwargs, _rows, mmct
    ):

        alt_args = [_rows // 25]
        alt_kwargs = {
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': [0, 2],
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': -1
        }

        # INITIALIZE->FIT_TRFM
        IFTCls = MCT(*alt_args, **alt_kwargs)
        SPFT_TRFM_X, SPFT_TRFM_Y = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT->SET_PARAMS->TRFM
        FSPTCls = MCT(*_args, **_kwargs)
        FSPTCls.fit(X.copy(), y.copy())
        FSPTCls.set_params(count_threshold=alt_args[0], **alt_kwargs)
        FSPT_TRFM_X, FSPT_TRFM_Y = FSPTCls.transform(X.copy(), y.copy())
        assert FSPTCls.count_threshold == alt_args[0]  # the og value

        # CHECK X AND Y EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)), \
            f"SPFT_TRFM_X != FSPT_TRFM_X"

        assert np.array_equiv(SPFT_TRFM_Y, FSPT_TRFM_Y), \
            f"SPFT_TRFM_Y != FSPT_TRFM_Y"


        # VERIFY transform AGAINST REFEREE OUTPUT WITH SAME INPUTS
        MOCK_X = mmct().trfm(
            X.copy(),
            None,
            alt_kwargs['ignore_columns'],
            alt_kwargs['ignore_nan'],
            alt_kwargs['ignore_non_binary_integer_columns'],
            alt_kwargs['ignore_float_columns'],
            alt_kwargs['handle_as_bool'],
            alt_kwargs['delete_axis_0'],
            alt_args[0]
        )

        assert np.array_equiv(FSPT_TRFM_X.astype(str), MOCK_X.astype(str)), \
            f"FSPT_TRFM_X != MOCK_X"

        del MOCK_X



    def test_set_params_between_fit_transforms(
        self, X, y, _args, _kwargs, _rows, mmct
    ):

        alt_args = [_rows // 25]
        alt_kwargs = {
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': [0, 2],
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': -1
        }


        # INITIALIZE->FIT_TRFM
        IFTCls = MCT(*alt_args, **alt_kwargs)
        SPFT_TRFM_X, SPFT_TRFM_Y = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT_TRFM->SET_PARAMS->FIT_TRFM
        FTSPFTCls = MCT(*alt_args, **alt_kwargs)
        FTSPFTCls.set_params(max_recursions=2)
        FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls.max_recursions == 2


        # pizza 25_01_27_10_16_00 MCT now blocks setting any params when
        # in fitted state with max_recursions >= 2
        with pytest.raises(ValueError):
            FTSPFTCls.set_params(**alt_kwargs)
        # FTSPFT_TRFM_X, FTSPFT_TRFM_Y = \
        #     FTSPFTCls.fit_transform(X.copy(), y.copy())
        #
        # assert FTSPFTCls.max_recursions == 1
        #
        # assert np.array_equiv(
        #     SPFT_TRFM_X.astype(str), FTSPFT_TRFM_X.astype(str)
        # ), \
        #     f"SPFT_TRFM_X != FTSPFT_TRFM_X"

        # assert np.array_equiv(SPFT_TRFM_Y, FTSPFT_TRFM_Y), \
        #     f"SPFT_TRFM_Y != FTSPFT_TRFM_Y"






















