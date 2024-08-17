# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from uuid import uuid4
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2

from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV




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
    def X_dask_array():
        return da.random.randint(0,10,(100, 30)).rechunk((20, 30))


    @staticmethod
    @pytest.fixture
    def X_COLUMNS(X_dask_array):
        return [str(uuid4())[:4] for _ in range(X_dask_array.shape[1])]


    @staticmethod
    @pytest.fixture
    def X_dask_df(X_dask_array, X_COLUMNS):
        return ddf.from_dask_array(X_dask_array, columns=X_COLUMNS)


    @staticmethod
    @pytest.fixture
    def X_dask_series(X_dask_df):
        return X_dask_df.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def X_np_array(X_dask_array):
        return X_dask_array.compute()


    @staticmethod
    @pytest.fixture
    def X_pd_df(X_dask_df):
        return X_dask_df.compute()


    @staticmethod
    @pytest.fixture
    def X_pd_series(X_pd_df):
        return X_pd_df.iloc[:, 0]


    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    y_dask_array = da.random.randint(0,10,(100, 1)).rechunk((20, 1))

    y_dask_df = ddf.from_dask_array(y_dask_array, columns=['y'])

    y_dask_series = y_dask_df.iloc[:, 0]

    y_np_array = y_dask_array.compute()

    y_pd_df = y_dask_df.compute()

    y_pd_series = y_pd_df.iloc[:, 0]

    # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_np_array(self, X_np_array, y):

        # y cannot be pd.DF, dask.DF for no ravel() attribute

        if isinstance(y,
            (pd.core.frame.DataFrame, ddf.core.DataFrame, ddf2.DataFrame)):

            with pytest.raises(AttributeError):
                dask_GridSearchCV(
                    dask_LogisticRegression(),
                    param_grid={}
                ).fit(X_np_array, y)
        else:
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_np_array, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_pd_df(self, X_pd_df, y):

        # NotImplementedError: Could not find signature for add_intercept: <DataFrame>

        with pytest.raises(NotImplementedError):
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_pd_df, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_pd_series(self, X_pd_series, y):
        # NotImplementedError: Could not find signature for add_intercept: <Series>

        with pytest.raises(NotImplementedError):
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_pd_series, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_array(self, X_dask_array, y):

        if isinstance(y,
            (pd.core.frame.DataFrame, ddf.core.DataFrame, ddf2.DataFrame)):
            
            # no ravel() attribute
            with pytest.raises(AttributeError):
                dask_GridSearchCV(
                    dask_LogisticRegression(),
                    param_grid={}
                ).fit(X_dask_array, y)
        else:
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_dask_array, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_df(self, X_dask_df, y):

        # NotImplementedError: Could not find signature for add_intercept

        with pytest.raises(NotImplementedError):
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_dask_df, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_series(self, X_dask_series, y):
        # NotImplementedError: Could not find signature for add_intercept

        with pytest.raises(NotImplementedError):
            dask_GridSearchCV(
                dask_LogisticRegression(),
                param_grid={}
            ).fit(X_dask_series, y)


























