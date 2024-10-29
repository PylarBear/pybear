# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from copy import deepcopy
import numpy as np

from pybear.preprocessing.ColumnDeduplicateTransformer. \
    ColumnDeduplicateTransformer import ColumnDeduplicateTransformer as CDT




@pytest.fixture(scope='function')
def _kwargs():
    return {
        'keep': 'first',
        'do_not_drop': None,
        'conflict': 'raise',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
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

        # DEFAULT KWARGS
        # _kwargs = {
        #     'keep': 'first',
        #     'do_not_drop': None,
        #     'conflict': 'raise',
        #     'rtol': 1e-5,
        #     'atol': 1e-8,
        #     'equal_nan': False,
        #     'n_jobs': -1
        # }


    def test_excepts_for_unknown_param(self, _kwargs):

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.set_params(garbage=1)


    def test_set_params_correctly_applies(self, X, y, _kwargs):

        TestCls = CDT(**_kwargs)

        # assert TestCls initiated correctly
        for _param, _value in _kwargs.items():
            assert getattr(TestCls, _param) == _value

        # set new params before fit
        _scd_params = deepcopy(_kwargs)
        _scd_params['keep'] = 'random'
        _scd_params['do_not_drop'] = [0]
        _scd_params['conflict'] = 'ignore'
        _scd_params['rtol'] = 1e-7
        _scd_params['atol'] = 1e-9
        _scd_params['equal_nan'] = True
        _scd_params['n_jobs'] = 2

        TestCls.set_params(**_scd_params)

        # assert new values set correctly
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


        # CDT doesnt have any hidden params (leading _ like _keep)
        # do a fit(), then assert all params are still set correctly
        TestCls.fit(X, y)

        # assert new values still set correctly
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


    def test_equality_set_params_before_and_after_fit(self, X, y, _kwargs):

        alt_kwargs = {
            'keep': 'last',
            'do_not_drop': [1],
            'conflict': 'raise',
            'rtol': 1e-3,
            'atol': 1e-5,
            'equal_nan': False,
            'n_jobs': 1
        }

        # INITIALIZE->FIT_TRFM
        IFTCls = CDT(**alt_kwargs)
        SPFT_TRFM_X = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT->SET_PARAMS->TRFM
        FSPTCls = CDT(**_kwargs)
        FSPTCls.fit(X.copy(), y.copy())
        FSPTCls.set_params(**alt_kwargs)
        FSPT_TRFM_X = FSPTCls.transform(X.copy())
        assert FSPTCls.keep == 'last'

        # CHECK X AND Y EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)), \
            f"SPFT_TRFM_X != FSPT_TRFM_X"


    def test_set_params_between_fit_transforms(self, X, y, _kwargs):

        alt_kwargs = {
            'keep': 'random',
            'do_not_drop': None,
            'conflict': 'raise',
            'rtol': 1e-7,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': -1
        }


        # INITIALIZE->FIT_TRFM
        IFTCls = CDT(**alt_kwargs)
        SPFT_TRFM_X = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT_TRFM->SET_PARAMS->FIT_TRFM
        FTSPFTCls = CDT(**alt_kwargs)
        FTSPFTCls.set_params(keep='first')
        FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls.keep == 'first'
        FTSPFTCls.set_params(**alt_kwargs)
        FTSPFT_TRFM_X = FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls.keep == 'random'

        assert np.array_equiv(
            SPFT_TRFM_X.astype(str), FTSPFT_TRFM_X.astype(str)
        ), \
            f"SPFT_TRFM_X != FTSPFT_TRFM_X"








