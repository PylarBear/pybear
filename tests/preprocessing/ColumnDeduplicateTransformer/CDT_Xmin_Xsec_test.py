# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer

import pytest

from uuid import uuid4
from copy import deepcopy
import itertools
import numpy as np
np.random.seed(1)
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
from dask_ml.wrappers import Incremental, ParallelPostFit





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

        if output_type == 'polars':
            pytest.skip(reason=f"skip testing polars output")

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
# @pytest.mark.skipif(bypass is True, reason=f"bypass")
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




# TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDaskIncrementalParallelPostFit:


    @staticmethod
    @pytest.fixture
    def MCT_not_wrapped(_kwargs):
        return CDT(**_kwargs)

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_parallel(_kwargs):
        return ParallelPostFit(CDT(**_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_incremental(_kwargs):
        return Incremental(CDT(**_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_both(_kwargs):
        return ParallelPostFit(Incremental(CDT(**_kwargs)))



    FORMATS = ['da', 'ddf_df', 'ddf_series', 'np', 'pddf', 'pdseries']
    @pytest.mark.parametrize('x_format', FORMATS)
    @pytest.mark.parametrize('y_format', FORMATS + [None])
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_always_fits_X_y_always_excepts_transform_with_y(self, wrappings,
        MCT_wrapped_parallel, MCT_wrapped_incremental, MCT_not_wrapped,
        MCT_wrapped_both, _dum_X, _columns, x_format, y_format, _kwargs,
        _shape
    ):

        # no difference with or without Client --- pizza

        # USE NUMERICAL COLUMNS ONLY 24_03_27_11_45_00  --- pizza
        # NotImplementedError: Cannot use auto rechunking with object dtype.
        # We are unable to estimate the size in bytes of object data

        if wrappings == 'incr':
            _test_cls = MCT_wrapped_parallel
        elif wrappings == 'ppf':
            _test_cls = MCT_wrapped_incremental
        elif wrappings == 'both':
            _test_cls = MCT_wrapped_both
        elif wrappings == 'none':
            _test_cls = MCT_not_wrapped

        _X = _dum_X.copy()
        _np_X = _dum_X.copy()
        _chunks = (_shape[0]//5, _shape[1])
        if x_format in ['pddf', 'pdseries']:
            _X = pd.DataFrame(data=_X, columns=_columns)
        if x_format == 'pdseries':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, pd.core.series.Series)
            _np_X = _X.to_frame().to_numpy()
        if x_format in ['da', 'ddf_df', 'ddf_series']:
            _X = da.from_array(_X, chunks=_chunks)
        if x_format in ['ddf_df', 'ddf_series']:
            _X = ddf.from_array(_X, chunksize=_chunks)
        if x_format == 'ddf_series':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, (ddf.core.Series, ddf2.Series))
            _np_X = _X.compute().to_frame().to_numpy()

        # confirm there is an X
        _X.shape


        y = np.random.randint(0,2,(_shape[0], 2))

        _y = y.copy()
        _np_y = _y.copy()
        _chunks = (_shape[0]//5, 2)
        if y_format in ['pddf', 'pdseries']:
            _y = pd.DataFrame(data=_y, columns=['y1', 'y2'])
        if y_format == 'pdseries':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, pd.core.series.Series)
            _np_y = _y.to_frame().to_numpy()
        if y_format in ['da', 'ddf_df', 'ddf_series']:
            _y = da.from_array(_y, chunks=_chunks)
        if y_format in ['ddf_df', 'ddf_series']:
            _y = ddf.from_array(_y, chunksize = _chunks)
        if y_format == 'ddf_series':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, (ddf.core.Series, ddf2.Series))
            _np_y = _y.compute().to_frame().to_numpy()
        if y_format is None:
            _y = None

        # confirm there is a y
        if _y is not None:
            _y.shape

        _was_fitted = False
        # incr covers fit() so should accept all objects for fits
        _dask = ['da', 'ddf_df', 'ddf_series']
        _non_dask = ['np', 'pddf', 'pdseries']

        a = x_format in _dask and y_format in _non_dask
        b = x_format in _non_dask and y_format in _dask
        if wrappings in ['ppf', 'both'] and (a + b) == 1:

            with pytest.raises(UnboundLocalError):
                _test_cls.partial_fit(_X, _y)

            with pytest.raises(UnboundLocalError):
                _test_cls.fit(_X, _y)

        else:
            _test_cls.partial_fit(_X, _y)
            _test_cls.fit(_X, _y)
            _was_fitted = True

        del _dask, _non_dask

        # ^^^ END fit ^^^

        # vvv transform vvv
        if _was_fitted:

            if x_format not in ['pdseries', 'ddf_series']:
                assert _X.shape[1] == _dum_X.shape[1]

            # always TypeError when try to pass y with ParallelPostFit
            _x_was_transformed = False
            _y_was_transformed = False
            if wrappings in ['ppf', 'both', 'incr']:

                with pytest.raises(TypeError):
                    _test_cls.transform(_X, _y)

                # always transforms with just X
                TRFM_X = _test_cls.transform(_X)
                _x_was_transformed = True

            elif wrappings in ['none']:

                if _y is not None:
                    _test_cls.transform(_X, _y)
                    TRFM_X, TRFM_Y = _test_cls.fit_transform(_X, _y)
                    _x_was_transformed = True
                    _y_was_transformed = True
                else:
                    _test_cls.transform(_X)
                    TRFM_X = _test_cls.fit_transform(_X, _y)
                    _x_was_transformed = True

            if _x_was_transformed:
                if x_format == 'np':
                    assert isinstance(TRFM_X, np.ndarray)
                if x_format == 'pddf':
                    assert isinstance(TRFM_X, pd.core.frame.DataFrame)
                if x_format == 'pdseries':
                    assert isinstance(TRFM_X, pd.core.series.Series)
                if x_format == 'da' and wrappings == 'none':
                    assert isinstance(TRFM_X, np.ndarray)
                elif x_format == 'da':
                    assert isinstance(TRFM_X, da.core.Array)
                if x_format == 'ddf_df' and wrappings == 'none':
                    assert isinstance(TRFM_X, pd.core.frame.DataFrame)
                elif x_format == 'ddf_df':
                    assert isinstance(TRFM_X, (ddf.core.DataFrame, ddf2.DataFrame))
                if x_format == 'ddf_series' and wrappings == 'none':
                    assert isinstance(TRFM_X, pd.core.series.Series)
                elif x_format == 'ddf_series':
                    assert isinstance(TRFM_X, (ddf.core.Series, ddf2.Series))

                if _y_was_transformed:
                    if y_format == 'np':
                        assert isinstance(TRFM_Y, np.ndarray)
                    if y_format == 'pddf':
                        assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
                    if y_format == 'pdseries':
                        assert isinstance(TRFM_Y, pd.core.series.Series)
                    if y_format == 'da' and wrappings == 'none':
                        assert isinstance(TRFM_Y, np.ndarray)
                    elif y_format == 'da':
                        assert isinstance(TRFM_Y, da.core.Array)
                    if y_format == 'ddf_df' and wrappings == 'none':
                        assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
                    elif y_format == 'ddf_df':
                        assert isinstance(TRFM_Y,
                            (ddf.core.DataFrame, ddf2.DataFrame)
                        )
                    if y_format == 'ddf_series' and wrappings == 'none':
                        assert isinstance(TRFM_Y, pd.core.series.Series)
                    elif y_format == 'ddf_series':
                        assert isinstance(TRFM_Y, (ddf.core.Series, ddf2.Series))

                # CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()
                try:
                    TRFM_X = TRFM_X.to_frame()
                except:
                    pass

                try:
                    TRFM_X = TRFM_X.compute()
                except:
                    pass

                try:
                    TRFM_X = TRFM_X.to_numpy()
                except:
                    pass

                if _y_was_transformed:

                    try:
                        TRFM_Y = TRFM_Y.to_frame()
                    except:
                        pass

                    try:
                        TRFM_Y = TRFM_Y.compute()
                    except:
                        pass

                    try:
                        TRFM_Y = TRFM_Y.to_numpy()
                    except:
                        pass

                # END CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()

                FitTransformTestCls = CDT(**_kwargs)
                if _y_was_transformed:
                    FT_TRFM_X, FT_TRFM_Y = \
                        FitTransformTestCls.fit_transform(_np_X, _np_y)
                else:
                    FT_TRFM_X = FitTransformTestCls.fit_transform(_np_X)

                assert isinstance(TRFM_X, np.ndarray)
                assert isinstance(FT_TRFM_X, np.ndarray)
                assert np.array_equiv(
                        TRFM_X.astype(str), FT_TRFM_X.astype(str)), \
                    (f"transformed X  != transformed np X on single fit/transform")

                if _y_was_transformed:
                    assert isinstance(TRFM_Y, np.ndarray)
                    assert isinstance(FT_TRFM_Y, np.ndarray)
                    assert np.array_equiv(
                        TRFM_Y.astype(str),
                        FT_TRFM_Y.astype(str)
                    ), f"transformed Y != transformed np Y on single fit/transform"

# END TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()


# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccuracyBeforeAndAfterFitAndTransform:

    @staticmethod
    def _attrs():
        return [
            'n_features_in_',
            'feature_names_in_',
            'duplicates_',
            'removed_columns_',
            'column_mask_'
        ]

    def test_attr_accuracy(
        self, _dum_X, _columns, _kwargs, _shape, _attrs
    ):

        NEW_X = _dum_X.copy()
        NEW_X_DF = pd.DataFrame(data=_dum_X, columns=_columns)

        NEW_Y = np.random.randint(0,2,_shape[0])
        NEW_Y_DF = pd.DataFrame(data=NEW_Y, _columns=['y'])

        # BEFORE FIT ***************************************************
        TestCls = CDT(**_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        for attr in _attrs:
            with pytest.raises(AttributeError):
                getattr(TestCls, attr)

        del TestCls
        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = CDT(**_kwargs)
            TestCls.fit(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ IF NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, _columns), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == _shape[0], \
                f"n_features_in_ after fit() != number of originally passed columns"

        del data_dtype, TEST_X, TEST_Y, TestCls

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        for data_dtype in ['np', 'pd']:

            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = CDT(**_kwargs)
            TestCls.fit_transform(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, _columns), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == _shape[1], \
                f"n_features_in_ after fit() != number of originally passed columns"

        del data_dtype, TEST_X, TEST_Y, TestCls
        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test1RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    def test_access_methods_before_fit(self, _dum_X, _kwargs):

        TestCls = CDT(**_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(NotFittedError):
            TestCls.test_threshold()

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # get_row_support()
        with pytest.raises(NotFittedError):
            TestCls.get_row_support(True)

        # get_support()
        with pytest.raises(NotFittedError):
            TestCls.get_support(True)

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(_dum_X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))
        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        # DEMONSTRATE EXCEPTS FOR UNKNOWN PARAM
        with pytest.raises(ValueError):
            TestCls.set_params(garbage=1)

        del TestCls, KEYS, VALUES, ATTRS

        TestCls = CDT(**_kwargs)
        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(_dum_X)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************

    def test_access_methods_after_fit(self, _X, _columns, _kwargs, _shape):

        X = _X()   # pizza figure out what this needs to be
        y = np.random.randint(0,2,_shape[0])

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = CDT(**_kwargs)
        TestCls.fit(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        TestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
        JUNK_ARGS = [float('inf'), np.pi, 'garbage', {'junk': 3},
                     [*range(len(_columns))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(TypeError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        _COLUMNS = np.array([f"x{i}" for i in range(len(_columns))])
        ACTIVE_COLUMNS = _COLUMNS[TestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv( TestCls.get_feature_names_out(None), ACTIVE_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")
        del ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out([f"x{i}" for i in range(2 * len(_columns))])

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")
        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE_COLUMNS

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = CDT(**_kwargs)
        TestCls.fit(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL COLUMNS
        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(TestCls.get_feature_names_out(None), _ACTIVE_COLUMNS), \
            f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE_COLUMNS

        # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features
        # FOR DISALLOWED TYPES

        JUNK_COL_NAMES = [
            [*range(len(_columns))], [*range(2 * len(_columns))], {'a': 1, 'b': 2}
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(TypeError):
                TestCls.get_feature_names_out(junk_col_names)

        del JUNK_COL_NAMES

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF input_features DOES
        # NOT EXACTLY MATCH ORIGINALLY FIT COLUMNS
        JUNK_COL_NAMES = [np.char.upper(_columns), np.hstack((_columns, _columns)), []]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        # WHEN HEADER PASSED TO (partial_)fit() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER

        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
                      np.array(_columns)[TestCls.get_support(False)]), \
            f"get_feature_names_out() did not return original columns"

        del junk_col_names, RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # get_row_support()
        with pytest.raises(AttributeError):
            TestCls.get_row_support(False)

        # get_support()
        for _indices in [True, False]:
            __ = TestCls.get_support(_indices)
            assert isinstance(__, np.ndarray), \
                f"get_support() did not return numpy.ndarray"

            if not _indices:
                assert __.dtype == 'bool', \
                    f"get_support with indices=False did not return a boolean array"
                assert len(__) == TestCls.n_features_in_, \
                    f"len(get_support(False)) != n_features_in_"
                assert sum(__) == len(TestCls.get_feature_names_out(None))
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_support with indices=True did not return an array of "
                     f"integers")
                assert len(__) == len(TestCls.get_feature_names_out(None))

        del TestCls, _indices, __,

        # ** _handle_X_y()

        # inverse_transform() ********************
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, y)  # X IS NP ARRAY

        # SHOULD RAISE ValueError IF X IS NOT A 2D ARRAY
        for junk_x in [[], [[]]]:
            with pytest.raises(ValueError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE TypeError IF X IS NOT A LIST-TYPE
        for junk_x in [None, 'junk_string', 3, np.pi]:
            with pytest.raises(TypeError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # RETAINED COLUMNS
        TRFM_X = TestCls.transform(X)
        TRFM_MASK = TestCls.get_support(False)
        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((
                            __[TRFM_MASK],
                            np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_columns)

                if diff_cols == 'same':
                    TestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match "
             f"originally fitted columns")

        __ = np.logical_not(TestCls.get_support(False))
        assert np.array_equiv(INV_TRFM_X[:, __],
                              np.zeros((TRFM_X.shape[0], sum(__)))
            ), \
            (f"back-filled parts of inverse_transform() output do not slice "
             f"to a zero array")
        del __

        assert np.array_equiv(
            TRFM_X.astype(str),
            INV_TRFM_X[:, TestCls.get_support(False)].astype(str)
            ), (f"output of inverse_transform() does not reduce back to "
                f"the output of transform()")

        del junk_x, TRFM_X, TRFM_MASK, obj_type, diff_cols
        del TEST_X, INV_TRFM_X, TestCls

        # END inverse_transform() **********

        TestCls = CDT(**_kwargs)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************

    def test_access_methods_after_transform(self, _X, _columns, _kwargs, _shape):

        X = _X()    # pizza figure out what this needs to be
        y = np.random.randint(0, 2, _shape[0])

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = CDT(**_kwargs).fit(X, y)
        TransformedTestCls = CDT(**_kwargs).fit(X, y)
        TRFM_X = TransformedTestCls.transform(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        # SHOULD BE THE SAME AS AFTER FIT
        TransformedTestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN ORIGINAL (UNSLICED) COLUMNS
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(_columns)


        _ACTIVE_COLUMNS = np.array(_columns)[TransformedTestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = CDT(**_kwargs)
        PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                      np.array(_columns)[PDTransformedTestCls.get_support(False)]), \
            (f"get_feature_names_out(None) after transform() != "
             f"originally passed columns")

        del PDTransformedTestCls
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TransformedTestCls.get_metadata_routing()

        # get_params()
        assert TransformedTestCls.get_params(True) == \
                FittedTestCls.get_params(True), \
            f"get_params() after transform() != before transform()"

        # get_row_support()
        for _indices in [True, False]:
            __ = TransformedTestCls.get_row_support(_indices)
            assert isinstance(__, np.ndarray), \
                f"get_row_support() did not return numpy.ndarray"

            if not _indices:
                assert __.dtype == 'bool', \
                    (f"get_row_support with indices=False did not return a "
                     f"boolean array")
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_row_support with indices=True did not return an "
                     f"array of integers")

        del __

        # get_support()
        assert np.array_equiv(FittedTestCls.get_support(False),
            TransformedTestCls.get_support(False)), \
            f"get_support(False) after transform() != get_support(False) before"

        # ** _handle_X_y()

        # inverse_transform() ************

        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)), \
            (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TransformedTestCls.set_output(transform='pandas_dataframe')
        TransformedTestCls.transform(X, y)

        del TransformedTestCls

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        TestCls.fit_transform(X, y)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del FittedTestCls, TestCls, TRFM_X

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS **
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test2RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    # CREATE AN INSTANCE WITH ONLY 1 RECURSION TO COMPARE 1X-TRFMED OBJECTS
    # AGAINST 2X-TRFMED OBJECTS
    @staticmethod
    @pytest.fixture
    def OneRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 1

        return CDT(**_kwargs)


    @staticmethod
    @pytest.fixture
    def TwoRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 2

        return CDT(**_kwargs)

    def test_before_fit_transform(self, OneRecurTestCls, TwoRecurTestCls,
        X, y, _args, _kwargs):

        # **************************************************************
        # vvv BEFORE fit_transform() vvv *******************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.test_threshold()

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_row_support(True)

        # get_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_support(True)

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.inverse_transform(X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]

        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        TwoRecurTestCls = CDT(**_kwargs)
        # transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE fit_transform() ^^^ ***************************
        # **************************************************************

        del TwoRecurTestCls


    def test_after_fit_transform(self, OneRecurTestCls, TwoRecurTestCls, _X,
        _columns, _kwargs, _shape):

        X = _X()   # pizza figure out what this needs to be
        y = np.random.randint(0,2,_shape[0])

        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(X, y)
        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # **************************************************************
        # vvv AFTER fit_transform() vvv ********************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        assert not np.array_equiv(ONE_RCR_TRFM_X, TWO_RCR_TRFM_X), \
            f"ONE_RCR_TRFM_X == TWO_RCR_TRFM_X when it shouldnt"

        TwoRecurTestCls.test_threshold(clean_printout=True)
        print(f'^^^ mask building instructions should be displayed above ^^^')


        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ['x0', ..., 'x(n-1)]
        _COLUMNS = np.array([f"x{i}" for i in range(len(_columns))])
        _ACTIVE_COLUMNS = _COLUMNS[TwoRecurTestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            _ACTIVE_COLUMNS
        ), (f"get_feature_names_out(None) after fit_transform() != sliced "
            f"array of generic headers"
        )
        del _ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) !=
        # n_features_in_
        with pytest.raises(ValueError):
            _COLUMNS = [f"x{i}" for i in range(2 * len(_columns))]
            TwoRecurTestCls.get_feature_names_out(_COLUMNS)
            del _COLUMNS

        # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"TwoRecur.get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
            np.array(_columns)[TwoRecurTestCls.get_support(False)]), \
            f"TwoRecur.get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = \
            OneRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = \
            TwoRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None:
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            np.array(_columns)[TwoRecurTestCls.get_support(False)]
            ), (f"TwoRecur.get_feature_names_out(None) after fit_transform() != "
            f"sliced originally passed columns"
        )

        # WHEN HEADER PASSED TO fit_transform() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but returned "
             f"{type(RETURNED_FROM_GFNO)}")
        assert np.array_equiv(
            RETURNED_FROM_GFNO,
            np.array(_columns)[TwoRecurTestCls.get_support(False)]
            ), f"get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^
        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        for _indices in [True, False]:
            _ONE = OneRecurTestCls.get_row_support(_indices)
            _TWO = TwoRecurTestCls.get_row_support(_indices)

            assert isinstance(_ONE, np.ndarray), \
                f"get_row_support() for 1 recursion did not return numpy.ndarray"
            assert isinstance(_TWO, np.ndarray), \
                f"get_row_support() for 2 recursions did not return numpy.ndarray"

            if not _indices:
                assert _ONE.dtype == 'bool', (f"get_row_support with indices=False "
                          f"for 1 recursion did not return a boolean array")
                assert _TWO.dtype == 'bool', (f"get_row_support with indices=False "
                              f"for 2 recursions did not return a boolean array")
                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF ROWS IN X
                assert len(_ONE) == _shape[0], \
                    (f"row_support vector length for 1 recursion != rows in "
                     f"passed data"
                )
                assert len(_TWO) == _shape[0], \
                    (f"row_support vector length for 2 recursions != rows in "
                     f"passed data"
                )
                # NUMBER OF Trues in ONE RECUR MUST == NUMBER OF ROWS IN
                # ONE RCR TRFM X; SAME FOR TWO RCR
                assert sum(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr Trues IN row_support != TRFM X rows"
                assert sum(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr Trues IN row_support != TRFM X rows"
                # NUMBER OF Trues IN ONE RECUR MUST BE >= NUMBER OF Trues
                # IN TWO RECUR
                assert sum(_ONE) >= sum(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # ANY Trues IN TWO RECUR MUST ALSO BE True IN ONE RECUR
                assert np.unique(_ONE[_TWO])[0] == True, \
                    (f"Rows that are to be kept in 2nd recur (True) were False "
                     f"in 1st recur")
            elif _indices:
                assert 'int' in str(_ONE.dtype).lower(), \
                    (f"get_row_support with indices=True for 1 recursion did not "
                     f"return an array of integers")
                assert 'int' in str(_TWO.dtype).lower(), \
                    (f"get_row_support with indices=True for 2 recursions did not "
                     f"return an array of integers")
                # len(row_support) ONE RECUR MUST == NUMBER OF ROWS IN ONE RCR
                # TRFM X; SAME FOR TWO RCR
                assert len(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr len(row_support) as idxs does not equal TRFM X rows"
                assert len(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr len(row_support) as idxs does not equal TRFM X rows "
                # NUMBER OF ROW IDXS IN ONE RECUR MUST BE >= NUM ROW IDXS IN TWO RECUR
                assert len(_ONE) >= len(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for row_idx in _TWO:
                    assert row_idx in _ONE, (f"Rows that are to be kept by 2nd "
                                             f"recur were not kept by 1st recur")

        del _ONE, _TWO, row_idx, _indices
        del ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y, TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y

        # get_support()
        for _indices in [True, False]:
            _ = OneRecurTestCls.get_support(_indices)
            __ = TwoRecurTestCls.get_support(_indices)
            assert isinstance(_, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"
            assert isinstance(__, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"

            if not _indices:
                assert _.dtype == 'bool', \
                    (f"1 recursion get_support with indices=False did not "
                     f"return a boolean array")
                assert __.dtype == 'bool', \
                    (f"2 recursion get_support with indices=False did not "
                     f"return a boolean array")

                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF COLUMNS IN X
                assert len(_) == _shape[1], \
                    f"1 recursion len(get_support({_indices})) != X columns"
                assert len(__) == _shape[1], \
                    f"2 recursion len(get_support({_indices})) != X columns"
                # NUM COLUMNS IN 1 RECURSION MUST BE <= NUM COLUMNS IN X
                assert sum(_) <= _shape[1], \
                    (f"impossibly, number of columns kept by 1 recursion > number "
                     f"of columns in X")
                # NUM COLUMNS IN 2 RECURSION MUST BE <= NUM COLUMNS IN 1 RECURSION
                assert sum(__) <= sum(_),\
                    (f"impossibly, number of columns kept by 2 recursion > number "
                     f"of columns kept by 1 recursion")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                assert np.unique(_[__])[0] == True, (f"Columns that are to be "
                         f"kept in 2nd recur (True) were False in 1st recur")
            elif _indices:
                assert 'int' in str(_.dtype).lower(), (f"1 recursion get_support "
                    f"with indices=True did not return an array of integers")
                assert 'int' in str(__.dtype).lower(), (f"2 recursion get_support "
                    f"with indices=True did not return an array of integers")
                # ONE RECURSION COLUMNS MUST BE <= n_features_in_
                assert len(_) <= _shape[1], \
                    f"impossibly, 1 recursion len(get_support({_indices})) > X columns"
                # TWO RECURSION COLUMNS MUST BE <= ONE RECURSION COLUMNS
                assert len(__) <= len(_), \
                    (f"2 recursion len(get_support({_indices})) > 1 "
                     f"recursion len(get_support({_indices}))")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for col_idx in __:
                    assert col_idx in _, (f"Columns that are to be kept by "
                              f"2nd recur were not kept by 1st recur")

        del TwoRecurTestCls, _, __, _indices, col_idx

        # ** _handle_X_y()

        # inverse_transform() ********************
        TwoRecurTestCls = CDT(**_kwargs)
        # X IS NP ARRAY
        TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF RETAINED COLUMNS
        __ = np.array(_columns)
        TRFM_MASK = TwoRecurTestCls.get_support(False)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__[TRFM_MASK],
                                              np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                        del _COLUMNS

                if diff_cols == 'same':
                    TwoRecurTestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TwoRecurTestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TwoRecurTestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TwoRecurTestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match originally "
             f"fitted columns")

        __ = np.logical_not(TwoRecurTestCls.get_support(False))
        _ZERO_ARRAY = np.zeros((TRFM_X.shape[0], sum(__)))
        assert np.array_equiv(INV_TRFM_X[:, __], _ZERO_ARRAY), (f"back-filled "
            f"parts of inverse_transform() output do not slice to a zero array")
        del __, _ZERO_ARRAY

        assert np.array_equiv(TRFM_X.astype(str),
            INV_TRFM_X[:, TwoRecurTestCls.get_support(False)].astype(str)), \
            (f"output of inverse_transform() does not reduce back to the output "
             f"of transform()")

        del TRFM_X, TRFM_Y, TRFM_MASK, obj_type, diff_cols, TEST_X, INV_TRFM_X

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'
        TwoRecurTestCls.fit_transform(X, y)
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'

        del TwoRecurTestCls

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del OneRecurTestCls,

        # END ^^^ AFTER fit_transform() ^^^ ****************************
        # **************************************************************


# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS


