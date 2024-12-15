# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import SlimPolyFeatures as SlimPoly

from pybear.utilities import nan_mask, nan_mask_numerical, nan_mask_string

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl
import dask.array as da
import dask.dataframe as ddf

import pytest






pytest.skip(reason=f"pizza not finished", allow_module_level=True)


bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _shape():
    return (10, 4)


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
        'equal_nan': False,
        'rtol': 1e-5,
        'atol': 1e-8,
        'n_jobs': 1     # leave this set at 1 because of confliction
    }


@pytest.fixture(scope='module')
def _X_np(_X_factory, _shape):
    return _X_factory(
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='function')
def _bad_columns(_master_columns, _shape):
    return _master_columns.copy()[-_shape[1]:]


@pytest.fixture(scope='module')
def _X_pd(_X_np, _columns):
    return pd.DataFrame(
        data=_X_np,
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


    # degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_degree',
        (None, [1,2], {1,2}, (1,2), {'a':1}, lambda x: x)
    )
    def test_junk_degree(self, _X_np, _kwargs, junk_degree):

        _kwargs['degree'] = junk_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_degree',
        (-1, 1, np.pi, True, False)
    )
    def test_bad_degree(self, _X_np, _kwargs, bad_degree):

        # degree lower bound of 2 is hard coded, so 1 is bad

        _kwargs['degree'] = bad_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_degree', (2,3))
    def test_good_degree(self, _X_pd, _columns, _kwargs, good_degree):

        _kwargs['degree'] = good_degree

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # min_degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_min_degree',
        (None, [1,2], {1,2}, (1,2), {'a':1}, lambda x: x)
    )
    def test_junk_min_degree(self, _X_np, _kwargs, junk_min_degree):

        _kwargs['min_degree'] = junk_min_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_min_degree',
        (-1, 0, np.pi, True, False)
    )
    def test_bad_min_degree(self, _X_np, _kwargs, bad_min_degree):

        # min_degree lower bound of 1 is hard coded, so 0 is bad

        _kwargs['min_degree'] = bad_min_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_min_degree', (2,3,4))
    def test_good_min_degree(self, _X_pd, _kwargs, good_min_degree):

        _kwargs['min_degree'] = good_min_degree
        _kwargs['degree'] = good_min_degree + 1

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END min_degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_keep',
        (True, False, None, [1,2], {1,2})
    )
    def test_junk_keep(self, _X_np, _kwargs, junk_keep):

        _kwargs['keep'] = junk_keep

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_keep',
        (-1, np.pi, 'rubbish', {1:'trash'}, lambda x: 'junk', min)
    )
    def test_bad_keep(self, _X_np, _kwargs, bad_keep):

        _kwargs['keep'] = bad_keep

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_keep', ('first', 'last', 'random'))
    def test_good_keep(self, _X_pd, _columns, _kwargs, good_keep):

        if good_keep == 'string':
            good_keep = _columns[0]

        _kwargs['keep'] = good_keep

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_interaction_only',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {1,2}, {'a':1}, lambda x: x)
    )
    def test_junk_interaction_only(self, _X_np, _kwargs, junk_interaction_only):

        _kwargs['interaction_only'] = junk_interaction_only

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_interaction_only', (True, False))
    def test_good_interaction_only(self, _X_pd, _columns, _kwargs, good_interaction_only):

        _kwargs['interaction_only'] = good_interaction_only

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # scan_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_scan_X',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {1,2}, {'a':1}, lambda x: x)
    )
    def test_junk_scan_X(self, _X_np, _kwargs, junk_scan_X):

        _kwargs['scan_X'] = junk_scan_X

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_scan_X', (True, False))
    def test_good_scan_X(self, _X_pd, _columns, _kwargs, good_scan_X):

        _kwargs['scan_X'] = good_scan_X

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END scan_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # sparse_output ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_sparse_output',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {1,2}, {'a':1}, lambda x: x)
    )
    def test_junk_sparse_output(self, _X_np, _kwargs, junk_sparse_output):

        _kwargs['sparse_output'] = junk_sparse_output

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_sparse_output', (True, False))
    def test_good_sparse_output(self, _X_pd, _columns, _kwargs, good_sparse_output):

        _kwargs['sparse_output'] = good_sparse_output

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END sparse_output ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # feature_name_combiner ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # can be Literal['as_indices', 'as_feature_names'] or Callable[[Iterable[str], tuple[int,...]], str]

    @pytest.mark.parametrize('junk_feature_name_combiner',
        (-2.7, -1, 0, 1, 2.7, True, False, None, (0,1), [1,2], {1,2}, {'a':1})
    )
    def test_junk_feature_name_combiner(self, _X_np, _kwargs, junk_feature_name_combiner):

        _kwargs['feature_name_combiner'] = junk_feature_name_combiner

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_feature_name_combiner',
        ('that', 'was', 'trash', lambda x: x, min, lambda x, y: x + y)
    )
    def test_bad_feature_name_combiner(self, _X_np, _kwargs, bad_feature_name_combiner):

        _kwargs['feature_name_combiner'] = bad_feature_name_combiner

        with pytest.raises(Exception):
            # this could except for numerous reasons. _val_feature_name_combiner tries to
            # pass feature names and a tuple of ints to the callable and sees if it
            # returns a str. the exception is whatever the callable is raising.
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_feature_name_combiner',
        ('as_indices', 'as_feature_names', lambda x, y: 'Column1')
    )
    def test_good_feature_name_combiner(self, _X_pd, _columns, _kwargs, good_feature_name_combiner):

        _kwargs['feature_name_combiner'] = good_feature_name_combiner

        SlimPoly(**_kwargs).fit_transform(_X_pd)
    # END feature_name_combiner ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_equal_nan',
        (-1, 0, 1, np.pi, None, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_non_bool_equal_nan(self, _X_np, _kwargs, junk_equal_nan):

        _kwargs['equal_nan'] = junk_equal_nan

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_equal_nan', [True, False])
    def test_equal_nan_accepts_bool(self, _X_np, _kwargs, good_equal_nan):

        _kwargs['equal_nan'] = good_equal_nan

        SlimPoly(**_kwargs).fit_transform(_X_np)

    # END equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_junk',
        (None, 'trash', [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_rtol_atol(self, _X_np, _kwargs, _trial, _junk):

        _kwargs[_trial] = _junk

        # non-num are handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_bad', [-np.pi, -2, -1, True, False])
    def test_bad_rtol_atol(self, _X_np, _kwargs, _trial, _bad):

        _kwargs[_trial] = _bad

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_good', (1e-5, 1e-6, 1e-1))
    def test_good_rtol_atol(self, _X_np, _kwargs, _trial, _good):

        _kwargs[_trial] = _good

        SlimPoly(**_kwargs).fit_transform(_X_np)

    # END rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, _X_np, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(self, _X_np, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, _X_np, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        SlimPoly(**_kwargs).fit_transform(_X_np)

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
    def test_fit(self, _kwargs, _X_np, _stuff):
        SlimPoly(**_kwargs).fit(_X_np, _stuff)

    @ pytest.mark.parametrize('_stuff', STUFF)
    def test_partial_fit(self, _kwargs, _X_np, _stuff):
        SlimPoly(**_kwargs).partial_fit(_X_np, _stuff)


# END ALWAYS ACCEPTS y==anything TO fit() AND partial_fit() #################


# TEST EXCEPTS ANYTIME X==None PASSED TO fit(), partial_fit(), AND transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsAnytimeXisNone:

    def test_excepts_anytime_x_is_none(self, _X_np, _kwargs):

        # this is handled by _val_X

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit(None)

        with pytest.raises(Exception):
            SlimPoly(**_kwargs).partial_fit(None)

        with pytest.raises(Exception):
            TestCls = SlimPoly(**_kwargs)
            TestCls.fit(_X_np)
            TestCls.transform(None)
            del TestCls

        with pytest.raises(Exception):
            SlimPoly(**_kwargs).fit_transform(None)


# END TEST EXCEPTS ANYTIME X==None PASSED TO fit(), partial_fit(), transform()


# VERIFY REJECTS X AS SINGLE COLUMN / SERIES ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestRejectsXAsSingleColumnOrSeries:

    # y is ignored

    @staticmethod
    @pytest.fixture(scope='module')
    def VECTOR_X(_X_np):
        return _X_np[:, 0].copy()


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


        with pytest.raises(Exception):
            # this is handled by sklearn.base.BaseEstimator._validate_data,
            # let it raise whatever
            SlimPoly(**_kwargs).fit_transform(_fst_fit_X)

# END VERIFY REJECTS X AS SINGLE COLUMN / SERIES ##############################



# VERIFY ACCEPTS SINGLE 2D COLUMN ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAccepts2Columns:

    # y is ignored

    @staticmethod
    @pytest.fixture(scope='module')
    def TWO_COLUMN_X(_X_np):
        return _X_np[:, :2].copy().reshape((-1, 2))


    @pytest.mark.parametrize('_fst_fit_x_format',
        ('numpy', 'pandas')
    )
    @pytest.mark.parametrize('_fst_fit_x_hdr', [True, None])
    def test_X_as_single_column(
        self, _kwargs, _columns, TWO_COLUMN_X, _fst_fit_x_format, _fst_fit_x_hdr
    ):

        if _fst_fit_x_format == 'numpy':
            if _fst_fit_x_hdr:
                pytest.skip(reason=f"numpy cannot have header")
            else:
                _fst_fit_X = TWO_COLUMN_X.copy()

        if _fst_fit_x_format == 'pandas':
            if _fst_fit_x_hdr:
                _fst_fit_X = pd.DataFrame(data=TWO_COLUMN_X, columns=_columns[:2])
            else:
                _fst_fit_X = pd.DataFrame(data=TWO_COLUMN_X)

        SlimPoly(**_kwargs).fit_transform(_fst_fit_X)

# END VERIFY ACCEPTS SINGLE 2D COLUMN ##################################


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

        TestCls = SlimPoly(**_kwargs)

        _factory_kwargs = {
            '_dupl':None, '_format':'pd', '_dtype':'flt', '_has_nan':False,
            '_constants': None, '_shape':_shape
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

# END TEST ValueError WHEN SEES A DF HEADER DIFFERENT FROM FIRST-SEEN HEADER


# TEST OUTPUT TYPES ####################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestOutputTypes:

    _base_objects = ['np_array', 'pandas', 'scipy_sparse_csc']

    @pytest.mark.parametrize('x_input_type', _base_objects)
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, _X_np, _columns, _kwargs, x_input_type, output_type
    ):


        if x_input_type == 'scipy_sparse_csc' and output_type == 'polars':
            pytest.skip(
                reason=f"sk cannot convert scipy sparse input to polars directly"
            )


        NEW_X = _X_np.copy()
        NEW_COLUMNS = _columns.copy()


        if x_input_type == 'np_array':
            TEST_X = NEW_X
        elif x_input_type == 'pandas':
            TEST_X = pd.DataFrame(data=NEW_X, columns=NEW_COLUMNS)
        elif x_input_type == 'scipy_sparse_csc':
            TEST_X = ss.csc_array(NEW_X)
        else:
            raise Exception

        TestCls = SlimPoly(**_kwargs)
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
        self, _X_np, _kwargs
    ):

        TestCls = SlimPoly(**_kwargs)
        TEST_X = _X_np.copy()

        # 1)
        for _ in range(5):
            TestCls.partial_fit(TEST_X)

        del TestCls

        # 2)
        TestCls = SlimPoly(**_kwargs)
        TestCls.fit(TEST_X)
        TestCls.partial_fit(TEST_X)

        del TestCls

        # 3)
        TestCls = SlimPoly(**_kwargs)
        TestCls.fit(TEST_X)
        TestCls.fit(TEST_X)

        del TestCls

        # 4) a call to fit() after a previous partial_fit() should be allowed
        TestCls = SlimPoly(**_kwargs)
        TestCls.partial_fit(TEST_X)
        TestCls.fit(TEST_X)

        # 5) fit transform should allow calls ad libido
        for _ in range(5):
            TestCls.fit_transform(TEST_X)

        del TEST_X, TestCls

# END TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###############


# TEST MANY PARTIAL FITS == ONE BIG FIT ********************************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestManyPartialFitsEqualOneBigFit:


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_many_partial_fits_equal_one_big_fit(
        self, _X_np, _kwargs, _shape, _keep, _equal_nan
    ):

        # **** **** **** **** **** **** **** **** **** **** **** **** ****
        # THIS TEST IS CRITICAL FOR VERIFYING THAT transform PULLS THE
        # SAME COLUMN INDICES FOR ALL CALLS TO transform() WHEN
        # keep=='random'
        # **** **** **** **** **** **** **** **** **** **** **** **** ****

        _kwargs['keep'] = _keep
        _kwargs['equal_nan'] = _equal_nan

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST THAT ONE-SHOT partial_fit/transform == ONE-SHOT fit/transform
        OneShotPartialFitTestCls = SlimPoly(**_kwargs)
        OneShotPartialFitTestCls.partial_fit(_X_np)

        OneShotFullFitTestCls = SlimPoly(**_kwargs)
        OneShotFullFitTestCls.fit(_X_np)

        # need to break this up and turn to strings because of nans...
        # _X_np _has_nan=False, but constants have a column of np.nans
        _ = OneShotPartialFitTestCls.expansion_combinations_
        __ = OneShotFullFitTestCls.expansion_combinations_
        assert _ == __
        del _, __

        ONE_SHOT_PARTIAL_FIT_TRFM_X = \
            OneShotPartialFitTestCls.transform(_X_np, copy=True)

        ONE_SHOT_FULL_FIT_TRFM_X = \
            OneShotFullFitTestCls.transform(_X_np, copy=True)

        assert ONE_SHOT_PARTIAL_FIT_TRFM_X.shape == \
               ONE_SHOT_FULL_FIT_TRFM_X.shape

        if _keep != 'random':
            # this has np.nan in it, convert to str
            assert np.array_equal(
                ONE_SHOT_PARTIAL_FIT_TRFM_X.astype(str),
                ONE_SHOT_FULL_FIT_TRFM_X.astype(str)
            ), f"one shot partial fit trfm X != one shot full fit trfm X"


        del ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_X

        # END TEST THAT ONE-SHOT partial_fit/transform==ONE-SHOT fit/transform
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST PARTIAL FIT KEPT COMBINATIONS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        SingleFitTestClass = SlimPoly(**_kwargs)
        SingleFitTestClass.fit(_X_np)
        _ = SingleFitTestClass.expansion_combinations_

        DoublePartialFitTestClass = SlimPoly(**_kwargs)
        DoublePartialFitTestClass.partial_fit(_X_np)
        __ = DoublePartialFitTestClass.expansion_combinations_
        DoublePartialFitTestClass.partial_fit(_X_np)
        ___ = DoublePartialFitTestClass.expansion_combinations_

        assert _ == __
        assert _ == ___

        del _, __, ___, SingleFitTestClass, DoublePartialFitTestClass

        # END PARTIAL FIT CONSTANTS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # TEST MANY PARTIAL FITS == ONE BIG FIT

        # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X
        _chunks = 5
        X_CHUNK_HOLDER = []
        for row_chunk in range(_chunks):
            _mask_start = row_chunk * _shape[0] // _chunks
            _mask_end = (row_chunk + 1) * _shape[0] // _chunks
            X_CHUNK_HOLDER.append(_X_np[_mask_start:_mask_end, :])
        del _mask_start, _mask_end

        assert np.array_equiv(
            np.vstack(X_CHUNK_HOLDER).astype(str), _X_np.astype(str)
        ), \
            f"agglomerated X chunks != original X"

        PartialFitTestCls = SlimPoly(**_kwargs)
        OneShotFitTransformTestCls = SlimPoly(**_kwargs)

        # PIECEMEAL PARTIAL FIT
        for X_CHUNK in X_CHUNK_HOLDER:
            PartialFitTestCls.partial_fit(X_CHUNK)

        # PIECEMEAL TRANSFORM ******************************************
        # THIS CANT BE UNDER THE partial_fit LOOP, ALL FITS MUST BE DONE
        # BEFORE DOING ANY TRFMS
        PARTIAL_TRFM_X_HOLDER = []
        for X_CHUNK in X_CHUNK_HOLDER:

            PARTIAL_TRFM_X_HOLDER.append(
                PartialFitTestCls.transform(X_CHUNK)
            )


        # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
        FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_X_HOLDER)
        # END PIECEMEAL TRANSFORM **************************************

        # DO ONE-SHOT TRANSFORM OF X ON THE PARTIALLY FIT INSTANCE
        FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM = \
            PartialFitTestCls.transform(_X_np)

        del PartialFitTestCls


        if _keep != 'random':

            # ONE-SHOT FIT TRANSFORM
            FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM = \
                OneShotFitTransformTestCls.fit_transform(_X_np)

            del OneShotFitTransformTestCls

            # ASSERT ALL AGGLOMERATED X TRFMS ARE EQUAL
            assert np.array_equiv(
                    FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                    FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str)
                ), \
                (f"compiled trfm X from partial fit / partial trfm != "
                 f"one-shot fit/trfm X")

            assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
                ), (f"trfm X from partial fits / one-shot trfm != one-shot "
                    f"fit/trfm X")

        elif _keep == 'random':

            assert np.array_equiv(
                    FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str),
                    FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
                ), (f"trfm X from partial fit / partial trfm != "
                 f"trfm X from partial fit / one-shot trfm/")


        # TEST MANY PARTIAL FITS == ONE BIG FIT
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **

# END TEST MANY PARTIAL FITS == ONE BIG FIT ****************************


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestConstantColumnsAccuracyOverManyPartialFits:


    # verify correct progression of reported constants as partial fits are done.
    # rig a set of arrays that have progressively decreasing constants


    @staticmethod
    @pytest.fixture()
    def _chunk_shape():
        return (50,20)


    @staticmethod
    @pytest.fixture()
    def _X(_X_factory, _chunk_shape):

        def foo(_format, _dtype, _has_nan, _constants, _noise):

            return _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _columns=None,
                _constants=_constants,
                _zeros=None,
                _shape=_chunk_shape
            )

        return foo


    @pytest.mark.parametrize('_format', ('np', ))
    @pytest.mark.parametrize('_dtype', ('int', 'flt', 'int', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 5))
    def test_accuracy(
        self, _kwargs, _X, _format, _dtype, _has_nan, _start_constants
    ):

        if _format not in ('np', 'pd') and _dtype not in ('flt', 'int'):
            pytest.skip(reason=f"cant put non-num in scipy sparse")


        _new_kwargs = deepcopy(_kwargs)
        _new_kwargs['equal_nan'] = True

        TestCls = SlimPoly(**_new_kwargs)

        # build a pool of non-constants to fill the constants in X along the way
        # build a starting data object for first partial fit, using full constants
        # build a y vector
        # do a verification partial_fit, assert reported constants for original X
        # make a holder for all the different _wip_Xs, to do one big fit at the end
        # for however many times u want to do this:
        #   randomly replace one of the constants with non-constant column
        #   partial_fit
        #   assert reported constants - should be one less (the randomly chosen)
        # at the very end, stack all the _wip_Xs, do one big fit, verify constants

        _pool_X = _X(_format, _dtype, _has_nan, None, _noise=1e-9)

        _wip_X = _X(_format, _dtype, _has_nan, _noise=1e-9)

        _y = np.random.randint(0, 2, _wip_X.shape[0])

        out = TestCls.partial_fit(_wip_X, _y).constant_columns_
        assert len(out) == len(_start_constants)
        for idx, v in _start_constants.items():
            if str(v) == 'nan':
                assert str(v) == str(_start_constants[idx])
            else:
                assert v == _start_constants[idx]

        # to know how many constants can come out, get total number of constants
        _const_pool = list(_start_constants)
        _num_consts = len(_const_pool)

        X_HOLDER = []
        X_HOLDER.append(_wip_X)

        # take out only half of the constants (arbitrary) v^v^v^v^v^v^v^v^v^v^v
        for trial in range(_num_consts//2):

            random_const = np.random.choice(_const_pool, 1, replace=False)[0]

            # take the random constant of out _start_constants and _const_pool,
            # and take a column out of the X pool to patch the constant in _wip_X
            _start_constants.pop(random_const)
            _const_pool.remove(random_const)

            _from_X = _wip_X[:, random_const]
            _from_pool = _pool_X[:, random_const]
            assert not np.array_equal(
                _from_X[np.logical_not(nan_mask(_from_X))],
                _from_pool[np.logical_not(nan_mask(_from_pool))]
            )

            del _from_X, _from_pool

            _wip_X[:, random_const] = _pool_X[:, random_const].copy()

            X_HOLDER.append(_wip_X)

            # verify correctly reported constants after this partial_fit!
            out = TestCls.partial_fit(_wip_X, _y).constant_columns_
            assert len(out) == len(_start_constants)
            for idx, v in _start_constants.items():
                if str(v) == 'nan':
                    assert str(v) == str(_start_constants[idx])
                else:
                    assert v == _start_constants[idx]

        # END take out only half of the constants (arbitrary) v^v^v^v^v^v^v^v^v



        # do a one-shot fit, compare results
        # stack all the _wip_Xs
        _final_X = np.vstack(X_HOLDER)

        out = SlimPoly(**_new_kwargs).fit(_final_X, _y).constant_columns_
        assert len(out) == len(_start_constants)
        for idx, v in _start_constants.items():
            if str(v) == 'nan':
                assert str(out[idx]) == str(_start_constants[idx])
            else:
                assert v == _start_constants[idx]



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

        out = SlimPoly(**_kwargs).fit_transform(_X)

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

        out = SlimPoly(**_kwargs).fit_transform(_X)

        assert np.array_equal(out[:, 0], _X[:, 0])
        assert all(nan_mask_numerical(out[:, -1]))




@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestPartialFit:

    #     def partial_fit(
    #         self,
    #         X: DataType,
    #         y: any=None
    #     ) -> Self:

    # - only accepts ndarray, pd.DataFrame, and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 2 columns
    # - allows nan
    # - validates all instance attrs --- not tested here, see _validation


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _junk_X, _kwargs):

        # this is being caught by _validation at the top of partial_fit.
        # in particular,
        # if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        #      hasattr(_X, 'toarray'):
        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).partial_fit(_junk_X)


    @pytest.mark.parametrize('_format',
         (
             'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
             'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
             'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
             'dask_array', 'dask_dataframe'
         )
    )
    def test_X_container(self, _X_np, _columns, _kwargs, _format):

        _X = _X_np.copy()

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
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
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
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        elif _format == 'dask_array':
            _X_wip = da.array(_X)
        elif _format == 'dask_dataframe':
            _ = da.array(_X)
            _X_wip = ddf.from_dask_array(_, columns=_columns)
        else:
            raise Exception

        if _format in ('dask_array', 'dask_dataframe'):
            with pytest.raises(TypeError):
                # handled by IM
                SlimPoly(**_kwargs).partial_fit(_X_wip)
        else:
            SlimPoly(**_kwargs).partial_fit(_X_wip)


    @pytest.mark.parametrize('_num_cols', (0, 1))
    def test_X_must_have_2_or_more_columns(self, _X_factory, _kwargs, _num_cols):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20, _num_cols)
        )[:, :_num_cols]

        _kwargs['keep'] = 'first'

        if _num_cols < 2:
            with pytest.raises(ValueError):
                SlimPoly(**_kwargs).partial_fit(_wip_X)
        else:
            SlimPoly(**_kwargs).partial_fit(_wip_X)


    def test_rejects_no_samples(self, _X_np, _kwargs, _columns):

        _X = _X_np.copy()

        # dont know what is actually catching this! maybe _validate_data?
        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(
                np.empty((0, _X.shape[1]), dtype=np.float64)
            )


    def test_rejects_1D(self, _X_factory, _kwargs):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20, 2)
        )

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(_wip_X[:, 0])



    @pytest.mark.parametrize('_dtype', ('str', 'obj'))
    def test_fit_transform_floats_as_str_dtypes(
        self, _X_factory, _dtype, _shape
    ):

        # make an array of floats....
        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants=None,
            _zeros=0,
            _shape=_shape
        )

        # set dtype
        _wip_X = _wip_X.astype('<U20' if _dtype == 'str' else object)

        _SPF = SlimPoly(
            keep='last',
            equal_nan=True,
            rtol=1e-5,
            atol=1e-8,
            n_jobs=1
        )

        out = _SPF.fit_transform(_wip_X)

        assert isinstance(out, np.ndarray)

        # keep == 'last'!
        _ref_column_mask = np.ones((_shape[1],)).astype(bool)
        MASK = [i in _constants for i in range(_shape[1])]
        _ref_column_mask[MASK] = False
        _ref_column_mask[sorted(list(_constants))[-1]] = True
        del MASK

        assert np.array_equal(_SPF.column_mask_, _ref_column_mask)


    # dont really need to test accuracy, see _partial_fit


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransform:

    #     def transform(
    #         self,
    #         X: DataType,
    #         *,
    #         copy: bool = None
    #     ) -> DataType:

    # - only accepts ndarray, pd.DataFrame, and all ss
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
    def test_copy_validation(self, _X_np, _shape, _kwargs, _copy):

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X_np)

        if isinstance(_copy, (bool, type(None))):
            _SPF.transform(_X_np, copy=_copy)
        else:
            with pytest.raises(TypeError):
                _SPF.transform(_X_np, copy=_copy)


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _X_np, _junk_X, _kwargs):

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X_np)

        # this is being caught by _validation at the top of transform.
        # in particular,
        # if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        #     hasattr(_X, 'toarray'):
        with pytest.raises(TypeError):
            _SPF.transform(_junk_X)


    @pytest.mark.parametrize('_format',
         (
             'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
             'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
             'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
             'dask_array', 'dask_dataframe'
         )
    )
    def test_X_container(self, _X_np, _columns, _kwargs, _format):

        _X = _X_np.copy()

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
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
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
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        elif _format == 'dask_array':
            _X_wip = da.array(_X)
        elif _format == 'dask_dataframe':
            _ = da.array(_X)
            _X_wip = ddf.from_dask_array(_, columns=_columns)
        else:
            raise Exception

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X)  # fit on numpy, not the converted data

        if _format in ('dask_array', 'dask_dataframe'):
            with pytest.raises(TypeError):
                # handled by IM
                _SPF.transform(_X_wip)
        else:
            out = _SPF.transform(_X_wip)
            assert isinstance(out, type(_X_wip))


    # test_X_must_have_2_or_more_columns(self)
    # this is dictated by partial_fit. partial_fit requires 2+ columns, and
    # transform must have same number of features as fit


    def test_rejects_no_samples(self, _X_np, _kwargs):

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X_np)

        # this is caught by if _X.shape[0] == 0 in _val_X
        with pytest.raises(ValueError):
            _SPF.transform(
                np.empty((0, _X_np.shape[1]), dtype=np.float64)
            )


    def test_rejects_1D(self, _X_factory, _kwargs):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=0,
            _shape=(20, 2)
        )

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_wip_X)

        with pytest.raises(ValueError):
            _SPF.transform(_wip_X[:, 0])


    def test_rejects_bad_num_features(self, _X_np, _kwargs, _columns):
        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # FITTED COLUMNS

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X_np)

        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = _X_np.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__)
                elif diff_cols == 'less':
                    TEST_X = _X_np[:, :-1].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[:-1])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((_X_np.copy(), _X_np.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__, np.char.upper(__)))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                else:
                    raise Exception

                if diff_cols == 'same':
                    _SPF.transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        _SPF.transform(TEST_X)

        del _SPF, obj_type, diff_cols, TEST_X

    # dont really need to test accuracy, see _transform














