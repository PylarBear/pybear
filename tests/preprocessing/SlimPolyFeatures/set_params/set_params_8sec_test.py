# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from copy import deepcopy
import numpy as np

from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import pytest





@pytest.fixture(scope='function')
def _kwargs():
    return {
        'degree': 2,
        'min_degree': 1,
        'keep': 'first',
        'interaction_only': False,
        'scan_X': False,
        'sparse_output': False,
        'feature_name_combiner': lambda _columns, _x: 'any old string',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
        'n_jobs': 1
    }


@pytest.fixture(scope='function')
def _rows():
    return 8


@pytest.fixture(scope='function')
def X(_rows):
    return np.random.randint(0, 10, (_rows, 5))


@pytest.fixture(scope='function')
def y(_rows):
    return np.random.randint(0, 2, (_rows, 2), dtype=np.uint8)




class TestSetParams:


    def test_rejects_bad_assignments_at_init(self, _kwargs):

        _junk_kwargs = deepcopy(_kwargs)
        _junk_kwargs['trash'] = 'junk'
        _junk_kwargs['garbage'] = 'waste'
        _junk_kwargs['refuse'] = 'rubbish'

        with pytest.raises(Exception):
            # pizza will be coming back to this!
            # this is managed by BaseEstimator, let it raise whatever
            SlimPoly(**_junk_kwargs)


    def test_rejects_bad_assignments_in_set_params(self, _kwargs):

        TestCls = SlimPoly(**_kwargs)

        _junk_kwargs = deepcopy(_kwargs)
        _junk_kwargs['trash'] = 'junk'
        _junk_kwargs['garbage'] = 'waste'
        _junk_kwargs['refuse'] = 'rubbish'

        with pytest.raises(Exception):
            # pizza will be coming back to this!
            # this is managed by BaseEstimator, let it raise whatever
            TestCls.set_params(**_junk_kwargs)


    def test_set_params_correctly_applies(self, X, y, _kwargs):

        TestCls = SlimPoly(**_kwargs)

        # assert TestCls initiated correctly after init
        for _param, _value in _kwargs.items():
            assert getattr(TestCls, _param) == _value

        # set new params before fit
        _scd_params = deepcopy(_kwargs)
        _scd_params['keep'] = 'random'
        _scd_params['rtol'] = 1e-7
        _scd_params['atol'] = 1e-9
        _scd_params['equal_nan'] = True
        _scd_params['n_jobs'] = 2

        TestCls.set_params(**_scd_params)

        # assert new values set correctly
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


        # do a fit(), then assert all params are still set correctly
        TestCls.fit(X, y)

        # assert new values still set correctly after fit
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value


    def test_blocks_some_params_after_fit(self, X, y):


        alt_kwargs = {
            'degree': 2,
            'min_degree': 1,
            'keep': 'first',
            'interaction_only': False,
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'any old string',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': 1
        }


        # INITIALIZE
        IFTCls = SlimPoly(**alt_kwargs)

        # CAN SET ANYTHING BEFORE FIT
        IFTCls.set_params(**alt_kwargs)

        # 'keep', 'sparse_output', 'feature_name_combiner',
        # and 'n_jobs' not blocked after fit()
        IFTCls.fit(X.copy(), y.copy())

        allowed_kwargs = {
            'keep': 'last',
            'sparse_output': True,
            'feature_name_combiner': lambda _columns, _X: 'whatever',
            'n_jobs': 2
        }

        IFTCls.set_params(**allowed_kwargs)

        for param, value in allowed_kwargs.items():
            assert getattr(IFTCls, param) == value


        #  ANYTHING OTHER THAN 'keep', 'sparse_output', 'feature_name_combiner',
        #         # and 'n_jobs' are blocked after fit() and warns.
        disallowed_kwargs = {
            'degree': 3,
            'min_degree': 2,
            'interaction_only': True,
            'scan_X': True,
            'rtol': 1e-3,
            'atol': 1e-5,
            'equal_nan': True,
        }

        for param, value in disallowed_kwargs.items():
            _og_value = getattr(IFTCls, param)
            with pytest.warns():
                IFTCls.set_params(param=value)
            # assert did not set the new value,
            # kept old, which was from alt_kwargs
            assert getattr(IFTCls, param) == _og_value == alt_kwargs[param]


    def test_equality_set_params_before_and_after_fit(self, X, y, _kwargs):

        # test the equality of the data output under:
        # 1) set_params -> fit -> transform
        # 2) fit -> set_params -> transform

        alt_kwargs = {
            'degree': 2,
            'min_degree': 1,
            'keep': 'last',
            'interaction_only': False,
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'any old string',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': 1
        }

        # INITIALIZE->FIT_TRFM
        IFTCls = SlimPoly(**alt_kwargs)
        SPFT_TRFM_X = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT->SET_PARAMS->TRFM
        FSPTCls = SlimPoly(**_kwargs)
        FSPTCls.fit(X.copy(), y.copy())
        # the only difference between _kwargs and alt_kwargs is 'keep'
        # 'keep' is the only thing being changed
        FSPTCls.set_params(**alt_kwargs)
        FSPT_TRFM_X = FSPTCls.transform(X.copy())
        assert FSPTCls.keep == 'last'

        # CHECK OUTPUT EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)), \
            f"SPFT_TRFM_X != FSPT_TRFM_X"


    def test_set_params_between_fit_transforms(self, X, y):

        alt_kwargs = {
            'degree': 3,
            'min_degree': 2,
            'keep': 'last',
            'interaction_only': True,
            'scan_X': True,
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'something old',
            'rtol': 1e-4,
            'atol': 1e-7,
            'equal_nan': True,
            'n_jobs': 1
        }


        # INITIALIZE->FIT_TRFM
        IFTCls = SlimPoly(**alt_kwargs)
        SPFT_TRFM_X = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT_TRFM->SET_PARAMS->FIT_TRFM
        FTSPFTCls = SlimPoly(**alt_kwargs)
        FTSPFTCls.set_params(keep='first')
        FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls.keep == 'first'
        FTSPFTCls.set_params(**alt_kwargs)
        FTSPFT_TRFM_X = FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls.keep == 'last'

        assert np.array_equiv(
            SPFT_TRFM_X.astype(str), FTSPFT_TRFM_X.astype(str)
        ), \
            f"SPFT_TRFM_X != FTSPFT_TRFM_X"


    def test_set_params_output_repeatability(self, _X_factory):

        # condition 1: prove that a changed instance gives correct result
        # initialize #1 with sparse_output=False, fit and transform, keep output
        # initialize #2 with sparse_output=True, fit.
        # use set_params on #2 to change sparse_output=False.
        # transform #2, and compare output with that of #1

        # condition 2: prove that changing and changing back gives same result
        # use set_params on #1 to change sparse_output to True.
        # do a transform()
        # use set_params on #1 again to change sparse_output to False.
        # transform #1 again, and compare with the first output

        _alt_kwargs = {
            'degree': 2,
            'min_degree': 1,
            'keep': 'first',
            'interaction_only': False,
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'something old',
            'rtol': 1e-4,
            'atol': 1e-7,
            'equal_nan': True,
            'n_jobs': 1
        }


        TEST_X = _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=False,
            _columns=None,
            _zeros=None,
            _shape=(20,3)
        )

        # first class: initialize, fit, transform, and keep result
        TestCls1 = SlimPoly(**_alt_kwargs).fit(TEST_X)
        FIRST_TRFM_X = TestCls1.transform(TEST_X, copy=True)

        # condition 1 ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # initialize #2 with sparse_output=True, fit.
        _dum_kwargs = deepcopy(_alt_kwargs)
        _dum_kwargs['sparse_output'] = True
        TestCls2 = SlimPoly(**_dum_kwargs).fit(TEST_X)
        # set different params and transform without fit
        # use set_params on #2 to change sparse_output to False.
        TestCls2.set_params(**_alt_kwargs)
        # transform #2, and compare output with that of #1
        COND_1_OUT = TestCls2.transform(TEST_X, copy=True)

        assert np.array_equal(COND_1_OUT, FIRST_TRFM_X)

        # END condition 1 ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # condition 2 ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # prove that changing and changing back gives same result

        # use set_params on #1 to change sparse_output to True.  DO NOT FIT!
        TestCls1.set_params(**_dum_kwargs)
        # do a transform()
        SECOND_TRFM_X = TestCls1.transform(TEST_X, copy=True)
        # should not be equal to first trfm with sparse_output = False
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # use set_params on #1 again to change sparse_output back to False.
        TestCls1.set_params(**_alt_kwargs)
        # transform #1 again, and compare with the first output
        THIRD_TRFM_X = TestCls1.transform(TEST_X, copy=True)

        assert np.array_equal(FIRST_TRFM_X, THIRD_TRFM_X)

        # END condition 2 ** * ** * ** * ** * ** * ** * ** * ** * ** *

        del TEST_X















