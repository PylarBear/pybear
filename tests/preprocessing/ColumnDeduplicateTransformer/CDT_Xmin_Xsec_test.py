# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl



# pizza come back and reread this, see if it needs anything!
# what about correct progression of reported duplicates as partial fits are done?
# how about a rigged set of vstacked arrays that have progressively decreasing duplicates?

# pytest.skip(reason=f"pizza is not done!", allow_module_level=True)

# PIZZA BE SURE TO TEST PD NA HANDLING! AND SPARSE!

# PIZZA! inverse_transform!

# PIZZA! TEST IN A PIPELINE!


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


# pizza is this even used
@pytest.fixture(scope='function')
def _std_dupl(_shape):
    return [[0, 4], [3, 5, _shape[1]-1]]


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
class TestInputValidation:

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
        _kwargs['do_not_drop'] = [v for i, v in enumerate(_columns) if i % 2 == 0]

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
        _kwargs['do_not_drop'] = [v for i, v in enumerate(_columns) if i % 2 == 0]

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


# TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), AND transform()
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

# END TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), OR transform()


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


# TEST OUTPUT TYPES ####################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestOutputTypes:

    _base_objects = ['np_array', 'pandas', 'scipy_sparse_csc']

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

# END TEST ALL COLUMNS THE SAME OR DIFFERENT #####################################


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test_Init_Fit_SetParams_Transform:

        # DEFAULT KWARGS
        # _kwargs = {
        #     'keep': 'first',
        #     'do_not_drop': None,
        #     'conflict': 'raise',
        #     'columns': None,
        #     'rtol': 1e-5,
        #     'atol': 1e-8,
        #     'equal_nan': False,
        #     'n_jobs': -1
        # }


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


    def test_assignments(self, _alt_kwargs):

        TestCls = CDT(**_alt_kwargs)

        # test after init
        for _alt_kwarg, value in _alt_kwargs.items():
            assert getattr(TestCls, _alt_kwarg) == value

        # test after setting different params
        _new_kwargs = deepcopy(_alt_kwargs)
        _new_kwargs['keep'] = 'random'
        _new_kwargs['conflict'] = 'ignore'
        _new_kwargs['n_jobs'] = 4
        TestCls.set_params(**_new_kwargs)

        # test after set_params
        for _new_kwarg, value in _new_kwargs.items():
            assert getattr(TestCls, _new_kwarg) == value


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


    def test_set_params(self, _X_factory, _alt_kwargs):

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

        assert np.array_equal(ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_X), \
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










