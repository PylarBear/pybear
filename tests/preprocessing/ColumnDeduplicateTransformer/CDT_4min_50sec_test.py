# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

from pybear.utilities import nan_mask, nan_mask_numerical, nan_mask_string

from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl
import dask.array as da
import dask.dataframe as ddf



bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _shape():
    return (20, 10)


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


@pytest.fixture(scope='module')
def _dum_X(_X_factory, _shape):
    return _X_factory(_dupl=None, _has_nan=False, _dtype='flt', _shape=_shape)


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='function')
def _bad_columns(_master_columns, _shape):
    return _master_columns.copy()[-_shape[1]:]


@pytest.fixture(scope='module')
def _X_pd(_dum_X, _columns):
    return pd.DataFrame(
        data=_dum_X,
        columns=_columns
)


# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInitValidation:

    JUNK = (
            -1,0,1, np.pi, True, False, None, 'trash', [1,2], {1,2}, {'a':1},
            lambda x: x, min
    )


    # keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_keep',
        (-1,0,1, np.pi, True, False, None, [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_keep(self, _dum_X, _kwargs, junk_keep):

        _kwargs['keep'] = junk_keep

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('bad_keep', ('trash', 'garbage', 'waste'))
    def test_bad_keep(self, _dum_X, _kwargs, bad_keep):

        _kwargs['keep'] = bad_keep

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('good_keep', ('first', 'last', 'random'))
    def test_good_keep(self, _dum_X, _kwargs, good_keep):

        _kwargs['keep'] = good_keep

        CDT(**_kwargs).fit_transform(_dum_X)
    # END keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # do_not_drop ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_type', ('np', 'pd'), scope='module')
    @pytest.mark.parametrize('_columns_is_passed', (True, False), scope='module')
    @pytest.mark.parametrize('junk_dnd',
        (-1, 0, 1, np.pi, True, False, 'trash', {'a': 1}, lambda x: x, min)
    )
    def test_rejects_not_list_like_or_none(
        self, _kwargs, _dum_X, _type, _columns, _columns_is_passed, junk_dnd
    ):

        if _type == 'np':
            _X = _dum_X
        else:
            _X = pd.DataFrame(
                data=_dum_X,
                columns=_columns if _columns_is_passed else None
            )

        _kwargs['do_not_drop'] = junk_dnd

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_X)


    @pytest.mark.parametrize('_type', ('np', 'pd'), scope='module')
    @pytest.mark.parametrize('_columns_is_passed', (True, False), scope='module')
    @pytest.mark.parametrize('bad_dnd',
        ([True, min, 3.14], [min, max, float], [2.718, 3.141, 8.834])
    )
    def test_rejects_bad_list(
        self, _dum_X, _kwargs, _type, _columns, _columns_is_passed, bad_dnd
    ):

        if _type == 'np':
            _X = _dum_X
        else:
            _X = pd.DataFrame(
                data=_dum_X,
                columns=_columns if _columns_is_passed else None
            )

        _kwargs['do_not_drop'] = bad_dnd

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_X)


    def test_array_str_handing(self, _dum_X, _kwargs, _columns):

        # rejects str when no header
        _kwargs['do_not_drop'] = \
            [v for i, v in enumerate(_columns) if i % 2 == 0]

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


        # rejects bad str when header
        _kwargs['do_not_drop'] = ['a', 'b']

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_array_int_and_none_handling(
            self, _dum_X, _kwargs, _columns_is_passed, _columns
    ):

        # accepts good int always
        _kwargs['do_not_drop'] = [0, 1]

        TestCls = CDT(**_kwargs)
        TestCls.fit_transform(_dum_X)


        # rejects bad int always - 1
        _kwargs['do_not_drop'] = [-1, 1]

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_dum_X)


        # rejects bad int always - 2
        _kwargs['do_not_drop'] = [0, _dum_X.shape[1]]

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_dum_X)


        # accepts None always
        _kwargs['do_not_drop'] = None

        TestCls = CDT(**_kwargs)
        TestCls.fit_transform(_dum_X)


    def test_df_str_handling(self, _X_pd, _kwargs, _columns):

        # accepts good str always
        _kwargs['do_not_drop'] = \
            [v for i, v in enumerate(_columns) if i % 2 == 0]

        TestCls = CDT(**_kwargs)
        TestCls.fit_transform(_X_pd)


        # rejects bad str always
        _kwargs['do_not_drop'] = ['a', 'b']

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_X_pd)


    def test_df_int_and_none_handling(self, _X_pd, _kwargs, _columns):
        # accepts good int always
        _kwargs['do_not_drop'] = [0, 1]

        TestCls = CDT(**_kwargs)
        TestCls.fit_transform(_X_pd)

        # rejects bad int always - 1
        _kwargs['do_not_drop'] = [-1, 1]

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_X_pd)

        # rejects bad int always - 2
        _kwargs['do_not_drop'] = [0, _X_pd.shape[1]]

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_X_pd)

        # columns can be None
        _kwargs['do_not_drop'] = None

        TestCls = CDT(**_kwargs)

        TestCls.fit_transform(_X_pd)
    # END do_not_drop ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # conflict  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_conflict',
        (-1, 0, np.pi, True, None, [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_conflict(self, _dum_X, _kwargs, junk_conflict):

        _kwargs['conflict'] = junk_conflict

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('bad_conflict', ('trash', 'garbage', 'waste'))
    def test_bad_conflict(self, _dum_X, _kwargs, bad_conflict):

        _kwargs['conflict'] = bad_conflict

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('good_conflict', ('raise', 'ignore'))
    def test_good_conflict(self, _dum_X, _kwargs, good_conflict):

        _kwargs['conflict'] = good_conflict

        CDT(**_kwargs).fit_transform(_dum_X)
    # END conflict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_junk',
        (True, False, None, 'trash', [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_rtol_atol(self, _dum_X, _kwargs, _trial, _junk):

        _kwargs[_trial] = _junk

        TestCls = CDT(**_kwargs)

        # except for bools, this is handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_bad', [-2, 0, 100_000_000])
    def test_bad_rtol_atol(self, _dum_X, _kwargs, _trial, _bad):

        _kwargs[_trial] = _bad

        TestCls = CDT(**_kwargs)

        # except for bools, this is handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_good', (1e-5, 1e-6, 1e-1))
    def test_good_rtol_atol(self, _dum_X, _kwargs, _trial, _good):

        _kwargs[_trial] = _good

        CDT(**_kwargs).fit_transform(_dum_X)

    # END rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_junk',
        (-1, 0, 1, np.pi, None, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_non_bool_equal_nan(self, _dum_X, _kwargs, _junk):

        _kwargs['equal_nan'] = _junk

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('_equal_nan', [True, False])
    def test_equal_nan_accepts_bool(self, _dum_X, _kwargs, _equal_nan):

        _kwargs['equal_nan'] = _equal_nan

        CDT(**_kwargs).fit_transform(_dum_X)

    # END equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, _dum_X, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        TestCls = CDT(**_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(self, _dum_X, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        TestCls = CDT(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_dum_X)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, _dum_X, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        CDT(**_kwargs).fit_transform(_dum_X)

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# END test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# ALWAYS ACCEPTS y==anything TO fit() AND partial_fit() #################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestFitPartialFitAcceptYEqualsAnything:

    STUFF = (
            -1,0,1, np.pi, True, False, None, 'trash', [1,2], {1,2}, {'a':1},
            lambda x: x, min
    )

    @pytest.mark.parametrize('_stuff', STUFF)
    def test_fit(self, _kwargs, _dum_X, _stuff):
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, _stuff)

    @ pytest.mark.parametrize('_stuff', STUFF)
    def test_partial_fit_after_partial_fit(self, _kwargs, _dum_X, _stuff):
        TestCls = CDT(**_kwargs)
        TestCls.partial_fit(_dum_X, _stuff)

    @ pytest.mark.parametrize('_stuff', STUFF)
    def test_partial_fit_after_fit(self, _kwargs, _dum_X, _stuff):
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, None)
        TestCls.partial_fit(_dum_X, _stuff)

# END ALWAYS ACCEPTS y==anything TO fit() AND partial_fit() #################


# TEST EXCEPTS ANYTIME X==None PASSED TO fit(), partial_fit(), AND transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsAnytimeXisNone:

    def test_excepts_anytime_x_is_none(self, _dum_X, _kwargs):

        # this is handled by sklearn.base.BaseEstimator._validate_data,
        # let it raise whatever

        with pytest.raises(Exception):
            TestCls = CDT(**_kwargs)
            TestCls.fit(None)

        with pytest.raises(Exception):
            TestCls = CDT(**_kwargs)
            TestCls.partial_fit(_dum_X)
            TestCls.partial_fit(None)

        with pytest.raises(Exception):
            TestCls = CDT(**_kwargs)
            TestCls.fit(_dum_X)
            TestCls.partial_fit(None)

        with pytest.raises(Exception):
            TestCls = CDT(**_kwargs)
            TestCls.fit(_dum_X)
            TestCls.transform(None)

        with pytest.raises(Exception):
            TestCls = CDT(**_kwargs)
            TestCls.fit_transform(None)

        del TestCls

# END TEST EXCEPTS ANYTIME X==None PASSED TO fit(), partial_fit(), transform()


# VERIFY REJECTS X AS SINGLE COLUMN / SERIES ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestRejectsXAsSingleColumnOrSeries:

    # y is ignored

    @staticmethod
    @pytest.fixture(scope='module')
    def VECTOR_X(_dum_X):
        return _dum_X[:, 0].copy()


    @pytest.mark.parametrize('_fst_fit_x_format',
        ('numpy', 'pandas_dataframe', 'pandas_series')
    )
    @pytest.mark.parametrize('_fst_fit_x_hdr', [True, None])
    def test_X_as_single_column(
        self, _kwargs, _columns, VECTOR_X, _fst_fit_x_format, _fst_fit_x_hdr
    ):

        if _fst_fit_x_format == 'numpy':
            if _fst_fit_x_hdr:
                pytest.skip(reason=f"numpy cannot have header")
            else:
                _fst_fit_X = VECTOR_X.copy()

        if 'pandas' in _fst_fit_x_format:
            if _fst_fit_x_hdr:
                _fst_fit_X = pd.DataFrame(data=VECTOR_X, columns=_columns[:1])
            else:
                _fst_fit_X = pd.DataFrame(data=VECTOR_X)

        # not elif!
        if _fst_fit_x_format == 'pandas_series':
            _fst_fit_X = _fst_fit_X.squeeze()

        TestCls = CDT(**_kwargs)

        with pytest.raises(Exception):
            # this is handled by sklearn.base.BaseEstimator._validate_data,
            # let it raise whatever
            TestCls.fit_transform(_fst_fit_X)

# END VERIFY REJECTS X AS SINGLE COLUMN / SERIES ##############################


# TEST ValueError WHEN SEES A DF HEADER DIFFERENT FROM FIRST-SEEN HEADER

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptWarnOnDifferentHeader:

    NAMES = ['DF1', 'DF2', 'NO_HDR_DF']

    @pytest.mark.parametrize('fst_fit_name', NAMES)
    @pytest.mark.parametrize('scd_fit_name', NAMES)
    @pytest.mark.parametrize('trfm_name', NAMES)
    def test_except_or_warn_on_different_headers(self, _X_factory, _kwargs,
        _columns, _bad_columns, fst_fit_name, scd_fit_name, trfm_name, _shape
    ):

        _col_dict = {'DF1': _columns, 'DF2': _bad_columns, 'NO_HDR_DF': None}

        TestCls = CDT(**_kwargs)

        _factory_kwargs = {
            '_dupl':None, '_format':'pd', '_dtype':'flt', '_has_nan':False,
            '_shape':_shape
        }

        fst_fit_X = _X_factory(_columns=_col_dict[fst_fit_name], **_factory_kwargs)
        scd_fit_X = _X_factory(_columns=_col_dict[scd_fit_name], **_factory_kwargs)
        trfm_X = _X_factory(_columns=_col_dict[trfm_name], **_factory_kwargs)

        _objs = [fst_fit_name, scd_fit_name, trfm_name]
        # EXCEPT IF 2 DIFFERENT HEADERS ARE SEEN
        sklearn_exception = 0
        sklearn_exception += bool('DF1' in _objs and 'DF2' in _objs)
        # IF FIRST FIT WAS WITH NO HEADER, THEN ANYTHING GETS THRU ON
        # SUBSEQUENT partial_fits AND transform
        sklearn_exception -= bool(fst_fit_name == 'NO_HDR_DF')
        sklearn_exception = max(0, sklearn_exception)

        # WARN IF HAS-HEADER AND NOT-HEADER BOTH PASSED DURING fits/transform
        sklearn_warn = 0
        if not sklearn_exception:
            sklearn_warn += ('NO_HDR_DF' in _objs and 'NO_HDR_DF' in _objs)
            # IF NONE OF THEM HAD A HEADER, THEN NO WARNING
            sklearn_warn -= ('DF1' not in _objs and 'DF2' not in _objs)
            sklearn_warn = max(0, sklearn_warn)

        del _objs

        if sklearn_exception:
            with pytest.raises(Exception):
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        elif sklearn_warn:
            with pytest.warns():
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        else:
            # SHOULD NOT EXCEPT OR WARN
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)

    del NAMES

# END TEST ValueError WHEN SEES A DF HEADER  DIFFERENT FROM FIRST-SEEN HEADER


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAllMethodsExceptOnScipyBSR:

    @pytest.mark.parametrize(f'_format', ('matrix', 'array'))
    def test_all_methods_reject_BSR(self, _format, _kwargs, _dum_X):

        TestCls = CDT(**deepcopy(_kwargs))

        if _format == 'matrix':
            BSR_X = ss.bsr_matrix(_dum_X)
        elif _format == 'array':
            BSR_X = ss.bsr_array(_dum_X)

        # sklearn _validate_data is not catching this!

        with pytest.raises(TypeError):
            TestCls.partial_fit(BSR_X)

        with pytest.raises(TypeError):
            TestCls.fit(BSR_X)

        TestCls.fit(_dum_X)

        with pytest.raises(TypeError):
            TestCls.transform(BSR_X)

        with pytest.raises(TypeError):
            TestCls.inverse_transform(BSR_X)


# TEST OUTPUT TYPES ####################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestOutputTypes:

    _base_objects = ('np_array', 'pandas', 'scipy_sparse_csc')

    @pytest.mark.parametrize('x_input_type', _base_objects)
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, _dum_X, _columns, _kwargs, x_input_type, output_type
    ):


        if x_input_type == 'scipy_sparse_csc' and output_type == 'polars':
            pytest.skip(
                reason=f"skcannot convert scipy sparse input to polars directly"
            )


        NEW_X = _dum_X.copy()
        NEW_COLUMNS = _columns.copy()


        if x_input_type == 'np_array':
            TEST_X = NEW_X
        elif x_input_type == 'pandas':
            TEST_X = pd.DataFrame(data=NEW_X, columns=NEW_COLUMNS)
        elif x_input_type == 'scipy_sparse_csc':
            TEST_X = ss.csc_array(NEW_X)
        else:
            raise Exception

        TestCls = CDT(**_kwargs)
        TestCls.set_output(transform=output_type)

        if x_input_type == 'scipy_sparse_csc' and output_type == 'pandas':
            # when passed a scipy sparse, sklearn cannot convert to
            # pandas output. this is a sklearn problem, let it raise whatever
            with pytest.raises(Exception):
                TRFM_X = TestCls.fit_transform(TEST_X)
            pytest.skip(reason=f"sklearn cannot convert csc to pandas")
        else:
            TRFM_X = TestCls.fit_transform(TEST_X)

        # if output_type is None, should return same type as given
        if output_type in [None, 'default']:
            assert type(TRFM_X) == type(TEST_X), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(TEST_X)})")
        # if output_type is 'default', should return np array no matter what given
        elif output_type == 'np_array':
            assert isinstance(TRFM_X, np.ndarray), \
                f"output_type is default or np_array, TRFM_X is {type(TRFM_X)}"
        # if output_type is 'pandas', should return pd df no matter what given
        elif output_type == 'pandas':
            # pandas.core.frame.DataFrame
            assert isinstance(TRFM_X, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
        elif output_type == 'polars':
            # polars.dataframe.frame.DataFrame
            assert isinstance(TRFM_X, pl.dataframe.frame.DataFrame), \
                f"output_type is polars dataframe, TRFM_X is {type(TRFM_X)}"
        else:
            raise Exception


# TEST OUTPUT TYPES ####################################################



# TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###################
# 1) partial_fit() should allow unlimited number of subsequent partial_fits()
# 2) one call to fit() should allow subsequent attempts to partial_fit()
# 3) one call to fit() should allow later attempts to fit() (2nd fit will reset)
# 4) calls to partial_fit() should allow later attempt to fit() (fit will reset)
# 5) fit_transform() should allow calls ad libido
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestConditionalAccessToPartialFitAndFit:

    def test_conditional_access_to_partial_fit_and_fit(
        self, _dum_X, _kwargs
    ):

        TestCls = CDT(**_kwargs)
        TEST_X = _dum_X.copy()

        # 1)
        for _ in range(5):
            TestCls.partial_fit(TEST_X)

        del TestCls

        # 2)
        TestCls = CDT(**_kwargs)
        TestCls.fit(TEST_X)
        TestCls.partial_fit(TEST_X)

        del TestCls

        # 3)
        TestCls = CDT(**_kwargs)
        TestCls.fit(TEST_X)
        TestCls.fit(TEST_X)

        del TestCls

        # 4) a call to fit() after a previous partial_fit() should be allowed
        TestCls = CDT(**_kwargs)
        TestCls.partial_fit(TEST_X)
        TestCls.fit(TEST_X)

        # 5) fit transform should allow calls ad libido
        for _ in range(5):
            TestCls.fit_transform(TEST_X)

        del TEST_X, TestCls

# END TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###############


# TEST ALL COLUMNS THE SAME OR DIFFERENT #####################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAllColumnsTheSameorDifferent:

    @pytest.mark.parametrize('same_or_diff', ('_same', '_diff'))
    @pytest.mark.parametrize('x_format', ('np', 'pd', 'coo'))
    def test_all_columns_the_same(
        self, _kwargs, _dum_X, same_or_diff, x_format, _columns, _shape
    ):

        TEST_X = _dum_X.copy()

        if same_or_diff == '_same':
            for col_idx in range(1, TEST_X.shape[1]):
                TEST_X[:, col_idx] = TEST_X[:, 0]

        if x_format == 'np':
            pass
        elif x_format == 'pd':
            TEST_X = pd.DataFrame(data=TEST_X, columns=_columns)
        elif x_format == 'coo':
            TEST_X = ss.coo_matrix(TEST_X)
        else:
            raise Exception

        TestCls = CDT(**_kwargs)
        out = TestCls.fit_transform(TEST_X)

        if same_or_diff == '_same':
            assert out.shape[1] == 1
        elif same_or_diff == '_diff':
            assert out.shape[1] == _shape[1]

# END TEST ALL COLUMNS THE SAME OR DIFFERENT ##################################


# TEST MANY PARTIAL FITS == ONE BIG FIT ********************************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestManyPartialFitsEqualOneBigFit:

    def test_many_partial_fits_equal_one_big_fit(
        self, _dum_X, _kwargs, _shape
    ):


        # ** ** ** ** ** ** ** ** ** ** **
        # TEST THAT ONE-SHOT partial_fit/transform == ONE-SHOT fit/transform
        OneShotPartialFitTestCls = CDT(**_kwargs)
        OneShotPartialFitTestCls.partial_fit(_dum_X)
        ONE_SHOT_PARTIAL_FIT_TRFM_X = \
            OneShotPartialFitTestCls.transform(_dum_X, copy=True)

        OneShotFullFitTestCls = CDT(**_kwargs)
        OneShotFullFitTestCls.partial_fit(_dum_X)
        ONE_SHOT_FULL_FIT_TRFM_X = \
            OneShotFullFitTestCls.transform(_dum_X, copy=True)

        assert np.array_equal(
            ONE_SHOT_PARTIAL_FIT_TRFM_X,
            ONE_SHOT_FULL_FIT_TRFM_X
        ), \
            f"one shot partial fit trfm X != one shot full fit trfm X"

        # END TEST THAT ONE-SHOT partial_fit/transform==ONE-SHOT fit/transform
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST PARTIAL FIT DUPLS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        SingleFitTestClass = CDT(**_kwargs)
        DoublePartialFitTestClass = CDT(**_kwargs)

        SingleFitTestClass.fit(_dum_X)
        DoublePartialFitTestClass.partial_fit(_dum_X)
        DoublePartialFitTestClass.partial_fit(_dum_X)

        # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X
        _chunks = 5
        X_CHUNK_HOLDER = []
        for row_chunk in range(_chunks):
            MASK1 = row_chunk * _shape[0] // _chunks
            MASK2 = (row_chunk + 1) * _shape[0] // _chunks
            X_CHUNK_HOLDER.append(_dum_X[MASK1:MASK2, :])
        del MASK1, MASK2

        assert np.array_equiv(
            np.vstack(X_CHUNK_HOLDER).astype(str), _dum_X.astype(str)
            ), \
            f"agglomerated X chunks != original X"

        PartialFitPartialTrfmTestCls = CDT(**_kwargs)
        PartialFitOneShotTrfmTestCls = CDT(**_kwargs)
        OneShotFitTransformTestCls = CDT(**_kwargs)

        # PIECEMEAL PARTIAL FIT
        for X_CHUNK in X_CHUNK_HOLDER:
            PartialFitPartialTrfmTestCls.partial_fit(X_CHUNK)
            PartialFitOneShotTrfmTestCls.partial_fit(X_CHUNK)

        # PIECEMEAL TRANSFORM ******************************************
        # THIS MUST BE IN ITS OWN LOOP, ALL FITS MUST BE DONE BEFORE
        # DOING ANY TRFMS
        PARTIAL_TRFM_X_HOLDER = []
        for X_CHUNK in X_CHUNK_HOLDER:
            PARTIAL_TRFM_X = \
                PartialFitPartialTrfmTestCls.transform(X_CHUNK)
            PARTIAL_TRFM_X_HOLDER.append(PARTIAL_TRFM_X)

        del PartialFitPartialTrfmTestCls, PARTIAL_TRFM_X

        # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
        FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_X_HOLDER)
        # END PIECEMEAL TRANSFORM **************************************

        # DO ONE-SHOT TRANSFORM OF X,y ON THE PARTIALLY FIT INSTANCE
        _ = PartialFitOneShotTrfmTestCls.transform(_dum_X)
        FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM = _
        del _

        # ONE-SHOT FIT TRANSFORM
        FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM = \
            OneShotFitTransformTestCls.fit_transform(_dum_X)

        # ASSERT ALL AGGLOMERATED X TRFMS ARE EQUAL
        assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str)
            ), \
            f"compiled trfm X from partial fit / partial trfm != one-shot fit/trfm X"

        assert np.array_equiv(
            FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
            FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
            ), f"trfm X from partial fits / one-shot trfm != one-shot fit/trfm X"

# END TEST MANY PARTIAL FITS == ONE BIG FIT ****************************


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDuplAccuracyOverManyPartialFits:


    # verify correct progression of reported duplicates as partial fits are done
    # rig a set of arrays that have progressively decreasing duplicates


    @staticmethod
    @pytest.fixture()
    def _chunk_shape():
        return (50,20)   # must have at least 10 columns for dupls to work


    @staticmethod
    @pytest.fixture()
    def _X(_X_factory, _chunk_shape):

        def foo(_dupl, _has_nan, _dtype):

            return _X_factory(
                _dupl=_dupl,
                _has_nan=_has_nan,
                _format='np',
                _dtype=_dtype,
                _columns=None,
                _zeros=None,
                _shape=_chunk_shape
            )

        return foo


    @staticmethod
    @pytest.fixture()
    def _start_dupl(_chunk_shape):
        # first indices of a set must be ascending
        return [
            [0, 7],
            [2, 4, _chunk_shape[1]-1],
            [3, 5, _chunk_shape[1]-2]
        ]



    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (0, 5))
    def test_accuracy(
        self, _kwargs, _X, _start_dupl, _has_nan, _dtype
    ):

        _new_kwargs = deepcopy(_kwargs)
        _new_kwargs['equal_nan'] = True

        TestCls = CDT(**_new_kwargs)

        # build a pool of non-dupls to fill the dupls in X along the way
        # build a starting data object for first partial fit, using full dupls
        # build a y vector
        # do a verification partial_fit, assert reported dupls for original X
        # make a holder for all the different _wip_Xs, to do one big fit at the end.
        # for however many times u want to do this:
        #   randomly replace one of the dupls with non-dupl column
        #   partial_fit
        #   assert reported dupls - should be one less (the randomly chosen column)
        # at the very end, stack all the _wip_Xs, do one big fit, verify dupls

        _pool_X = _X(None , _has_nan, _dtype)

        _wip_X = _X(_start_dupl, _has_nan, _dtype)

        _y = np.random.randint(0, 2, _wip_X.shape[0])

        out = TestCls.partial_fit(_wip_X, _y).duplicates_
        assert len(out) == len(_start_dupl)
        for idx in range(len(_start_dupl)):
            assert np.array_equal(out[idx], _start_dupl[idx])

        # to know how many dupls we can take out, get the total number of dupls
        _dupl_pool = list(itertools.chain(*_start_dupl))
        _num_dupls = len(_dupl_pool)

        X_HOLDER = []
        X_HOLDER.append(_wip_X)

        # take out only half of the dupls (arbitrary) v^v^v^v^v^v^v^v^v^v^v^v^
        for trial in range(_num_dupls//2):

            random_dupl = np.random.choice(_dupl_pool, 1, replace=False)[0]

            # take the random dupl of out _start_dupl and _dupl_pool, and take
            # a column out of the X pool to patch the dupl in _wip_X

            for _idx, _set in enumerate(reversed(_start_dupl)):
                try:
                    _start_dupl[_idx].remove(random_dupl)
                    if len(_start_dupl[_idx]) == 1:
                        # gotta take that single dupl out of dupl pool!
                        _dupl_pool.remove(_start_dupl[_idx][0])
                        del _start_dupl[_idx]
                    break
                except:
                    continue
            else:
                raise Exception(f"could not find dupl idx in _start_dupl")

            _dupl_pool.remove(random_dupl)


            _from_X = _wip_X[:, random_dupl]
            _from_pool = _pool_X[:, random_dupl]
            assert not np.array_equal(
                _from_X[np.logical_not(nan_mask(_from_X))],
                _from_pool[np.logical_not(nan_mask(_from_pool))]
            )

            _wip_X[:, random_dupl] = _pool_X[:, random_dupl].copy()

            X_HOLDER.append(_wip_X)

            # verify correctly reported dupls after this partial_fit!
            out = TestCls.partial_fit(_wip_X, _y).duplicates_
            assert len(out) == len(_start_dupl)
            for idx in range(len(_start_dupl)):
                assert np.array_equal(out[idx], _start_dupl[idx])

        # END take out only half of the dupls (arbitrary) v^v^v^v^v^v^v^v^v^v^v



        # do a one-shot fit, compare results
        # stack all the _wip_Xs
        _final_X = np.vstack(X_HOLDER)

        out = CDT(**_new_kwargs).fit(_final_X, _y).duplicates_
        assert len(out) == len(_start_dupl)
        for idx in range(len(_start_dupl)):
            assert np.array_equal(out[idx], _start_dupl[idx])



@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAColumnOfAllNans:

    def test_one_all_nans(self, _X_factory, _kwargs, _shape):

        _X = _X_factory(
            _dupl=[[0,1]],
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=(_shape[0], 3)
        )

        # set a column to all nans
        _X[:, -1] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))

        # 2nd column should drop, should have 2 columns, last is all np.nan

        TestCls = CDT(**_kwargs)
        out = TestCls.fit_transform(_X)

        assert np.array_equal(out[:, 0], _X[:, 0])
        assert all(nan_mask_numerical(out[:, -1]))


    def test_two_all_nans(self, _X_factory, _kwargs, _shape):

        _X = _X_factory(
            _dupl=[[0,1]],
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=(_shape[0], 4)
        )

        # set last 2 columns to all nans
        _X[:, [-2, -1]] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))
        assert all(nan_mask_numerical(_X[:, -2]))

        # 2nd & 4th column should drop, should have 2 columns, last is all np.nan

        TestCls = CDT(**_kwargs)
        out = TestCls.fit_transform(_X)

        assert np.array_equal(out[:, 0], _X[:, 0])
        assert all(nan_mask_numerical(out[:, -1]))



@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestPartialFit:
    
    #     def partial_fit(
    #         self,
    #         X: DataType,
    #         y: any=None
    #     ) -> Self:

    # - only accepts ndarray, pd.DataFrame, and all ss except BSR
    # - cannot be None
    # - must be 2D
    # - must have at least 2 columns
    # - allows nan
    # - validates all instance attrs --- not tested here, see _validation


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _junk_X, _kwargs):

        _CDT = CDT(**_kwargs)

        # this is being caught by _validation at the top of partial_fit.
        # in particular,
        # if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        #      hasattr(_X, 'toarray'):
        with pytest.raises(TypeError):
            _CDT.partial_fit(_junk_X)


    @pytest.mark.parametrize('_format',
    (
         'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
         'dia_array', 'lil_array', 'dok_array', 'dask_array', 'dask_dataframe'
    )
    )
    def test_X_container(self, _dum_X, _columns, _kwargs, _format):

        # dont need to test if rejects scipy BSR, see TestAllMethodsExceptOnScipyBSR

        _X = _dum_X.copy()

        if _format == 'np':
            _X_wip = _X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_columns
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'dask_array':
            _X_wip = da.array(_X)
        elif _format == 'dask_dataframe':
            _ = da.array(_X)
            _X_wip = ddf.from_dask_array(_, columns=_columns)
        else:
            raise Exception

        _CDT = CDT(**_kwargs)

        if _format in ('dask_array', 'dask_dataframe'):
            with pytest.raises(TypeError):
                # handled by CDT
                _CDT.partial_fit(_X_wip)
        else:
            _CDT.partial_fit(_X_wip)


    @pytest.mark.parametrize('_num_cols', (1, 2))
    def test_X_must_have_2_or_more_columns(self, _X_factory, _kwargs, _num_cols):

        # CDT transform() CAN *NEVER* REDUCE A DATASET TO ZERO COLUMNS,
        # ALWAYS MUST BE AT LEAST ONE LEFT

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20, 2)
        )[:, np.arange(_num_cols)]

        _kwargs['keep'] = 'first'
        _CDT = CDT(**_kwargs)

        if _num_cols < 2:
            with pytest.raises(ValueError):
                _CDT.partial_fit(_wip_X)
        else:
            _CDT.partial_fit(_wip_X)


    def test_rejects_no_samples(self, _dum_X, _kwargs, _columns):

        _X = _dum_X.copy()

        _CDT = CDT(**_kwargs)

        # dont know what is actually catching this! maybe _validate_data?
        with pytest.raises(ValueError):
            _CDT.partial_fit(np.empty((0, _X.shape[1]), dtype=np.float64))


    def test_rejects_1D(self, _X_factory, _kwargs):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20,2)
        )

        _CDT = CDT(**_kwargs)

        with pytest.raises(ValueError):
            _CDT.partial_fit(_wip_X[:, 0])


    # dont really need to test accuracy, see _partial_fit



@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransform:

    #     def transform(
    #         self,
    #         X: DataType,
    #         *,
    #         copy: bool = None
    #     ) -> DataType:

    # - only accepts ndarray, pd.DataFrame, and all ss except BSR
    # - cannot be None
    # - must be 2D
    # - must have at least 2 columns
    # - must have at least 1 sample
    # - num columns must equal num columns seen during fit
    # - allows nan
    # - output is C contiguous
    # - validates all instance attrs -- this isnt tested here, see _validation


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0, 1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _dum_X, _shape, _kwargs, _copy):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        if isinstance(_copy, (bool, type(None))):
            _CDT.transform(_dum_X, copy=_copy)
        else:
            with pytest.raises(TypeError):
                _CDT.transform(_dum_X, copy=_copy)


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _dum_X, _junk_X, _kwargs):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        # this is being caught by _validation at the top of transform.
        # in particular,
        # if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        #     hasattr(_X, 'toarray'):
        with pytest.raises(TypeError):
            _CDT.transform(_junk_X)


    @pytest.mark.parametrize('_format',
    (
         'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
         'dia_array', 'lil_array', 'dok_array', 'dask_array', 'dask_dataframe'
    )
    )
    def test_X_container(self, _dum_X, _columns, _kwargs, _format):

        # dont need to test if rejects scipy BSR, see TestAllMethodsExceptOnScipyBSR

        _X = _dum_X.copy()

        if _format == 'np':
            _X_wip = _X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_columns
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'dask_array':
            _X_wip = da.array(_X)
        elif _format == 'dask_dataframe':
            _ = da.array(_X)
            _X_wip = ddf.from_dask_array(_, columns=_columns)
        else:
            raise Exception

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X)    # fit on numpy, not the converted data

        if _format in ('dask_array', 'dask_dataframe'):
            with pytest.raises(TypeError):
                # handled by CDT
                _CDT.transform(_X_wip)
        else:
            _CDT.transform(_X_wip)


    # test_X_must_have_2_or_more_columns
    # this is dictated by partial_fit. partial_fit requires 2+ columns, and
    # transform must have same number of features as fit


    def test_rejects_no_samples(self, _dum_X, _kwargs):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        # this is caught by if _X.shape[0] == 0 in _val_X
        with pytest.raises(ValueError):
            _CDT.transform(
                np.empty((0, np.sum(_CDT.column_mask_)),dtype=np.float64)
            )


    def test_rejects_1D(self, _X_factory, _kwargs):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20,2)
        )

        _CDT = CDT(**_kwargs)
        _CDT.fit(_wip_X)

        with pytest.raises(ValueError):
            _CDT.transform(_wip_X[:, 0])


    def test_rejects_bad_num_features(self, _dum_X, _kwargs, _columns):
        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # FITTED COLUMNS

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        TRFM_X = _CDT.transform(_dum_X)
        TRFM_MASK = _CDT.column_mask_
        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :-1].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(
                            data=TEST_X,
                            columns=__[TRFM_MASK][:-1]
                        )
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((
                            __[TRFM_MASK],
                            np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                else:
                    raise Exception

                if diff_cols == 'same':
                    _CDT.transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        _CDT.transform(TEST_X)

        del _CDT, TRFM_X, TRFM_MASK, obj_type, diff_cols, TEST_X


    # dont really need to test accuracy, see _transform



@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInverseTransform:

    #     def inverse_transform(
    #         self,
    #         X: DataType,
    #         *,
    #         copy: bool = None
    #         ) -> DataType:

    # - only accepts ndarray, pd.DataFrame, and all ss except BSR
    # - cannot be None
    # - must be 2D
    # - must have at least 1 column
    # - must have at least 1 sample
    # - num columns must equal num columns in column_mask_
    # - allows nan
    # - output is C contiguous
    # - no instance attr validation

    # if we cant come up with a way to validate/ensure that the data
    # being inverted back matches with the current state of the params,
    # then will have to put some disclaimers in the docs.


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0,1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _dum_X, _shape, _kwargs, _copy):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        if isinstance(_copy, (bool, type(None))):
            _CDT.inverse_transform(_dum_X[:, _CDT.column_mask_], copy=_copy)
        else:
            with pytest.raises(TypeError):
                _CDT.inverse_transform(_dum_X[:, _CDT.column_mask_], copy=_copy)


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _dum_X, _junk_X, _kwargs):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        # this is being caught by _val_X at the top of inverse_transform.
        # in particular,
        # if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        #     hasattr(_X, 'toarray'):
        with pytest.raises(TypeError):
            _CDT.inverse_transform(_junk_X)


    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
            'dia_array', 'lil_array', 'dok_array', 'dask_array', 'dask_dataframe'
        )
    )
    def test_X_container(self, _dum_X, _columns, _kwargs, _format):

        # dont need to test if rejects scipy BSR, see TestAllMethodsExceptOnScipyBSR

        _X = _dum_X.copy()

        if _format == 'np':
            _X_wip = _X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_columns
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'dask_array':
            _X_wip = da.array(_X)
        elif _format == 'dask_dataframe':
            _ = da.array(_X)
            _X_wip = ddf.from_dask_array(_, columns=_columns)
        else:
            raise Exception

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X)  # fit on numpy, not the converted data

        if _format == 'dask_array':
            with pytest.raises(TypeError):
                # handled by CDT
                _CDT.inverse_transform(_X_wip[:, _CDT.column_mask_])
        elif _format == 'dask_dataframe':
            with pytest.raises(TypeError):
                # handled by CDT
                _CDT.inverse_transform(_X_wip.iloc[:, _CDT.column_mask_])
        elif _format == 'pd':
            _CDT.inverse_transform(_X_wip.iloc[:, _CDT.column_mask_])
        elif _format == 'np':
            _CDT.inverse_transform(_X_wip[:, _CDT.column_mask_])
        elif hasattr(_X_wip, 'toarray'):
            _og_dtype = type(_X_wip)
            _CDT.inverse_transform(_og_dtype(_X_wip.tocsc()[:, _CDT.column_mask_]))
            del _og_dtype
        else:
            raise Exception


    @pytest.mark.parametrize('_dim', ('0D', '1D'))
    def test_X_must_be_2D(self, _X_factory, _kwargs, _dim):

        # CDT transform() CAN *NEVER* REDUCE A DATASET TO ZERO COLUMNS,
        # ALWAYS MUST BE AT LEAST ONE LEFT. ZERO-D PROVES inverse_transform
        # REJECTS LESS THAN 1 COLUMN.

        _wip_X = _X_factory(
            _dupl=[[0, 1, 2]],
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20, 3)
        )

        _kwargs['keep'] = 'first'
        _CDT = CDT(**_kwargs)
        TRFM_X = _CDT.fit_transform(_wip_X)
        # _wip_X is rigged to transform to only one column
        assert TRFM_X.shape[1] == 1

        if _dim == '0D':
            TEST_TRFM_X = np.delete(
                TRFM_X,
                np.arange(TRFM_X.shape[1]),
                axis=1
            )
        elif _dim == '1D':
            TEST_TRFM_X = TRFM_X.ravel()
        else:
            raise Exception

        with pytest.raises(ValueError):
            _CDT.inverse_transform(TEST_TRFM_X)


    def test_rejects_no_samples(self, _dum_X, _kwargs):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        # this is caught by if _X.shape[0] == 0 in _val_X
        with pytest.raises(ValueError):
            _CDT.inverse_transform(
                np.empty((0, np.sum(_CDT.column_mask_)), dtype=np.float64)
            )


    def test_rejects_bad_num_features(self, _dum_X, _kwargs, _columns):
        # RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # COLUMNS RETAINED BY column_mask_

        _CDT = CDT(**_kwargs)
        _CDT.fit(_dum_X)

        TRFM_X = _CDT.transform(_dum_X)
        TRFM_MASK = _CDT.column_mask_
        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :-1].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(
                            data=TEST_X,
                            columns=__[TRFM_MASK][:-1]
                        )
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((
                            __[TRFM_MASK],
                            np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                else:
                    raise Exception

                if diff_cols == 'same':
                    _CDT.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        _CDT.inverse_transform(TEST_X)

        del _CDT, TRFM_X, TRFM_MASK, obj_type, diff_cols, TEST_X


    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
            'dia_array', 'lil_array', 'dok_array'
        )
    )
    @pytest.mark.parametrize('_dtype', ('int', 'flt', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_copy', (True, False))
    def test_accuracy(
        self, _X_factory, _columns, _kwargs, _shape, _format, _dtype,
        _has_nan, _keep, _copy
    ):

        # may not need to test accuracy here, see _inverse_transform,
        # but it is pretty straightforward. affirms the CDT class
        # inverse_transform method works correctly, above and beyond just
        # the _inverse_transform function called within.

        # set_output does not control the output container for inverse_transform
        # the output container is always the same as passed

        # dont need to test if rejects scipy BSR, see TestAllMethodsExceptOnScipyBSR

        if _format not in ('np', 'pd') and _dtype not in ('int', 'flt'):
            pytest.skip(reason=f"scipy sparse cant take non-numeric")

        _base_X = _X_factory(
            _dupl=None,
            _has_nan=_has_nan,
            _format='np',
            _dtype=_dtype,
            _columns=_columns,
            _zeros=0,
            _shape=_shape
        )


        if _format == 'np':
            _X_wip = _base_X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=_columns
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_base_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_base_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_base_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_base_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_base_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_base_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_base_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_base_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_base_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_base_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_base_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_base_X)
        else:
            raise Exception

        _kwargs['keep'] = _keep

        _CDT = CDT(**_kwargs)

        # fit v v v v v v v v v v v v v v v v v v v v
        _CDT.fit(_X_wip)
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        TRFM_X = _CDT.transform(_X_wip, copy=True)
        assert isinstance(TRFM_X, type(_X_wip))
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # inverse transform v v v v v v v v v v v v v v v
        INV_TRFM_X = _CDT.inverse_transform(
            X=TRFM_X,
            copy=_copy
        )
        assert isinstance(INV_TRFM_X, type(_X_wip))
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # output constainer is same as passed
        assert isinstance(INV_TRFM_X, type(_X_wip))

        # if output is numpy, order is C
        if isinstance(INV_TRFM_X, np.ndarray):
            assert INV_TRFM_X.flags['C_CONTIGUOUS'] is True

        # verify dimension of output
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == _CDT.n_features_in_, \
            (f"num features in output of inverse_transform() do not match "
             f"originally fitted columns")

        # convert everything to ndarray to use array_equal
        if isinstance(TRFM_X, np.ndarray):
            NP_TRFM_X = TRFM_X
            NP_INV_TRFM_X = INV_TRFM_X
        elif isinstance(TRFM_X, pd.core.frame.DataFrame):
            NP_TRFM_X = TRFM_X.to_numpy()
            NP_INV_TRFM_X = INV_TRFM_X.to_numpy()
        elif hasattr(TRFM_X, 'toarray'):
            NP_TRFM_X = TRFM_X.toarray()
            NP_INV_TRFM_X = INV_TRFM_X.toarray()
        else:
            raise Exception

        assert isinstance(NP_TRFM_X, np.ndarray)
        assert isinstance(NP_INV_TRFM_X, np.ndarray)

        # assert output is equal to originally transformed data
        # manage equal_nan for num or str
        try:
            NP_INV_TRFM_X.astype(np.float64)
            # when num
            assert np.array_equal(NP_INV_TRFM_X, _base_X, equal_nan=True), \
                f"inverse transform of transformed data does not equal original data"

            assert np.array_equal(
                NP_TRFM_X,
                NP_INV_TRFM_X[:, _CDT.column_mask_],
                equal_nan=True
            ), \
                (f"output of inverse_transform() does not reduce back to the "
                 f"output of transform()")
        except:
            # when str
            # CDT converts all nan-likes to np.nan, need to standardize them
            # in the inputs here for array_equal against the outputs
            # for string data, array_equal does not accept equal_nan param
            _base_X[nan_mask_string(_base_X)] = 'nan'
            NP_TRFM_X[nan_mask_string(NP_TRFM_X)] = 'nan'
            NP_INV_TRFM_X[nan_mask_string(NP_INV_TRFM_X)] = 'nan'

            assert np.array_equal(NP_INV_TRFM_X, _base_X), \
                f"inverse transform of transformed data != original data"

            assert np.array_equal(
                NP_TRFM_X,
                NP_INV_TRFM_X[:, _CDT.column_mask_]
            ), \
                (f"output of inverse_transform() does not reduce back to the "
                 f"output of transform()")






















