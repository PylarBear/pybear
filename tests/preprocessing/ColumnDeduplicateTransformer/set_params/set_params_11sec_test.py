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
        'n_jobs': -1        # confliction isnt a problem, 1 or -1 is fine
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


    def test_rejects_bad_assignments_at_init(self, _alt_kwargs):

        _junk_kwargs = deepcopy(_alt_kwargs)
        _junk_kwargs['trash'] = 'junk'
        _junk_kwargs['garbage'] = 'waste'
        _junk_kwargs['refuse'] = 'rubbish'

        with pytest.raises(Exception):
            # this is managed by BaseEstimator, let it raise whatever
            CDT(**_junk_kwargs)


    def test_rejects_bad_assignments_in_set_params(self, _alt_kwargs):
        TestCls = CDT(**_alt_kwargs)

        _junk_kwargs = deepcopy(_alt_kwargs)
        _junk_kwargs['trash'] = 'junk'
        _junk_kwargs['garbage'] = 'waste'
        _junk_kwargs['refuse'] = 'rubbish'

        with pytest.raises(Exception):
            # this is managed by BaseEstimator, let it raise whatever
            TestCls.set_params(**_junk_kwargs)


    def test_set_params_correctly_applies(self, X, y, _kwargs):

        TestCls = CDT(**_kwargs)

        # assert TestCls initiated correctly after init
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

        # assert new values still set correctly after fit
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


    def test_equality_set_params_before_and_after_fit(self, X, y, _kwargs):

        # test the equality of the data output under:
        # 1) set_params -> fit -> transform
        # 2) fit -> set_params -> transform

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



    @staticmethod
    @pytest.fixture()
    def _alt_kwargs():
        return {
            'keep': 'first',
            'do_not_drop': None,
            'conflict': 'raise',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': True,
            'n_jobs': -1
        }


    def test_set_params_output_repeatability(self, _X_factory, _alt_kwargs):

        # rig data so that it is 3 columns, first and last are duplicates

        # condition 1: prove that a changed instance gives correct result
        # initialize #1 with keep='first', fit and transform, keep output
        # initialize #2 with keep='last', fit.
        # use set_params on #2 to change keep to 'first'.
        # transform #2, and compare output with that of #1

        # condition 2: prove that changing and changing back gives same result
        # use set_params on #1 to change keep to 'last'.
        # do a transform()
        # use set_params on #1 again to change keep to 'first'.
        # transform #1 again, and compare with the first output

        _dupl = [[0,2]]

        TEST_X = _X_factory(
            _dupl=_dupl,
            _format='np',
            _dtype='flt',
            _has_nan=False,
            _columns=None,
            _zeros=None,
            _shape=(20,3)
        )

        # first class: initialize, fit, transform, and keep result
        TestCls1 = CDT(**_alt_kwargs).fit(TEST_X)
        FIRST_TRFM_X = TestCls1.transform(TEST_X, copy=True)

        # condition 1 ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # initialize #2 with keep='last', fit.
        _dum_kwargs = deepcopy(_alt_kwargs)
        _dum_kwargs['keep'] = 'last'
        TestCls2 = CDT(**_dum_kwargs)
        TestCls2.fit(TEST_X)
        # set different params and transform without fit
        # use set_params on #2 to change keep to 'first'.
        TestCls2.set_params(**_alt_kwargs)
        # transform #2, and compare output with that of #1
        COND_1_OUT = TestCls2.transform(TEST_X, copy=True)


        assert np.array_equal(COND_1_OUT, FIRST_TRFM_X)
        # since kept 'first', OUT should be TEST_X[:, :2]
        assert np.array_equal(COND_1_OUT, TEST_X[:, :2])

        # END condition 1 ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # condition 2 ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # prove that changing and changing back gives same result

        # use set_params on #1 to change keep to 'last'.  DO NOT FIT!
        TestCls1.set_params(**_dum_kwargs)
        # do a transform()
        SECOND_TRFM_X = TestCls1.transform(TEST_X, copy=True)
        # should not be equal to first trfm with keep='first'
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)
        # kept 'last' when 0 & 2 of 0,1,2 were identical, should leave 1,2
        assert np.array_equal(SECOND_TRFM_X, TEST_X[:, [1, 2]])

        # use set_params on #1 again to change keep to 'first'.
        TestCls1.set_params(**_alt_kwargs)
        # transform #1 again, and compare with the first output
        THIRD_TRFM_X = TestCls1.transform(TEST_X, copy=True)

        assert np.array_equal(FIRST_TRFM_X, THIRD_TRFM_X)

        # kept 'last' when 0 & 2 of 0,1,2 were identical, should leave 1,2
        assert np.array_equal(SECOND_TRFM_X, TEST_X[:, [1, 2]])

        # END condition 2 ** * ** * ** * ** * ** * ** * ** * ** * ** *

        del TEST_X




