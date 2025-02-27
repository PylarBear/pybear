# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import pandas as pd
import dask.dataframe as ddf


pytest.skip(reason=f'for edification purposes only', allow_module_level=True)
# no need to test the volatile and unpredictable state of dask error messages

# do not pass _client!  much slower for some reason

class TestDaskLogisticWrappedWithDaskGSCV:

    # AS OF 24_06_28_12_09_00 only fit dask_Logistic WITH X & Y ARE BOTH
    # NP_ARRAY OR DASK_ARRAY.

    # 24_06_28_11_39_00 The simplest distillation is:
    # 1) cannot ever take pd nor dask DFs for X, no add_intercept attr
    # 2) cannot ever take pd nor dask Series for X, no add_intercept attr
    # 3) cannot ever take pd nor dask DFs for y, no ravel attr
    # 4) OK to mix dask and non-dask objects, dask GSCV must be rechunking

    # so the way to robustly pass X and y to dask GSCV is as numpy array
    # or dask array
    #
    # IN SKLEARN, WHEN pd.DF IS PASSED TO sklearn_GridSearchCV.fit():
    #  -- "n_features_in_" BECOMES AVAILABLE
    #  -- "feature_names_in_" BECOMES AVAILABLE
    # IN DASK, WHEN ddf.DF OR pd.DF IS PASSED TO dask_GridSearchCV.fit():
    #  -- "n_features_in_" IS NEVER MADE AVAILABLE
    #  -- "feature_names_in_" IS NEVER MADE AVAILABLE
    #
    # REMEMBER dask_Logistic CAN ONLY fit() ARRAYS, SO NO FUNCTIONALITY
    # IS LOST IF A dask.ddf IS STRIPPED DOWN TO A dask.array.

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def X_dask_series(X_ddf):
        return X_ddf.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def X_pd_series(X_pd):
        return X_pd.iloc[:, 0]


    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def y_dask_series(y_ddf):
        return y_ddf.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def y_pd_series(y_pd):
        return y_pd.iloc[:, 0]

    # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_y_format',
        ('array', 'pd_df', 'pd_series', 'dask_array', 'dask_df', 'dask_series')
    )
    def test_passing_X_y_in_different_formats(self,
        dask_GSCV_est_log_one_scorer_prefit, X_da, X_ddf, X_dask_series,
        X_np, X_pd, X_pd_series, _y_format, y_da, y_ddf, y_dask_series,
        y_np, y_pd, y_pd_series
    ):

        if _y_format == 'array':
            _y = y_np
        elif _y_format == 'pd_df':
            _y = y_pd
        elif _y_format == 'pd_series':
            _y = y_pd_series
        elif _y_format == 'dask_array':
            _y = y_da
        elif _y_format == 'dask_df':
            _y = y_ddf
        elif _y_format == 'dask_series':
            _y = y_dask_series

    # make a copy of the prefit GSCV since the fixture is session scope, the
    # below fits would stick to it

        shallow_params = dask_GSCV_est_log_one_scorer_prefit.get_params(deep=False)
        deep_params = dask_GSCV_est_log_one_scorer_prefit.get_params(deep=True)
        dask_GSCV = type(dask_GSCV_est_log_one_scorer_prefit)(**shallow_params)
        dask_GSCV.set_params(**deep_params)

    # test_np_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # y cannot be pd.DF, dask.DF for no ravel() attribute

        if isinstance(_y, (pd.core.frame.DataFrame, ddf.core.DataFrame)):

            with pytest.raises(AttributeError):
                dask_GSCV.fit(X_np, _y)
        else:
            dask_GSCV.fit(X_np, _y)
    # END test_np_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # test_pd_df ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # AttributeError: 'DataFrame' object has no attribute 'ravel'
        if _y_format in ['pd_df', 'dask_df']:
            with pytest.raises(AttributeError):
                dask_GSCV.fit(X_pd, _y)
        else:
            dask_GSCV.fit(X_pd, _y)

    # END test_pd_df ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # test_pd_series ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ValueError: Expected a 2-dimensional container but got
    # <class 'pandas.core.series.Series'> instead. Pass a DataFrame
    # containing a single row (i.e. single sample) or a single column
    # (i.e. single feature) instead.

        with pytest.raises(ValueError):
            dask_GSCV.fit(X_pd_series, _y)

    # END test_pd_series ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # test_dask_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if isinstance(_y, (pd.core.frame.DataFrame, ddf.core.DataFrame)):

            # no ravel() attribute
            with pytest.raises(AttributeError):
                dask_GSCV.fit(X_da, _y)
        else:
            dask_GSCV.fit(X_da, _y)

    # END test_dask_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # test_dask_df ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # AttributeError: 'DataFrame' object has no attribute 'ravel'

        if _y_format in ['pd_df', 'dask_df']:
            with pytest.raises(AttributeError):
                dask_GSCV.fit(X_ddf, _y)
        else:
            dask_GSCV.fit(X_ddf, _y)

    # END test_dask_df ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # test_dask_series ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # NotImplementedError: Could not find signature for add_intercept

    # ValueError: Expected a 2-dimensional container but got
    # <class 'pandas.core.series.Series'> instead. Pass a DataFrame
    # containing a single row (i.e. single sample) or a single column
    # (i.e. single feature) instead.

        with pytest.raises(ValueError):
            dask_GSCV.fit(X_dask_series, _y)

    # END test_dask_series ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
























