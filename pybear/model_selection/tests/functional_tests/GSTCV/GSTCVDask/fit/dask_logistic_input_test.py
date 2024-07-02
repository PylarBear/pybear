# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import inspect

from uuid import uuid4
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf

from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV




class TestDaskLogistic:


    # 24_06_28_11_39_00 there are a lot of ifs ands and buts about what
    # dask_Logistic.fit() can take for X and y, but the simplest distillation
    # is:
    # 1) cannot ever take pd nor dask DFs for X, no add_intercept attr
    # 2) cannot ever take pd nor dask Series for X, no add_intercept attr
    # 3) cannot ever take pd nor dask DFs for y, no ravel attr
    # 4) cannot mix dask and non-dask Xs & ys due to chunk mismatches

    # so the simplest way to robustly pass X and y to logistic fit is
    # both are either numpy array or dask array

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def X_np_array():
        return np.random.randint(0,10,(100, 30))


    @staticmethod
    @pytest.fixture
    def X_COLUMNS(X_np_array):
        return [str(uuid4())[:4] for _ in range(X_np_array.shape[1])]


    @staticmethod
    @pytest.fixture
    def X_pd_df(X_np_array, X_COLUMNS):
        return pd.DataFrame(data=X_np_array, columns=X_COLUMNS)


    @staticmethod
    @pytest.fixture
    def X_pd_series(X_pd_df):
        return X_pd_df.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def X_dask_array(X_np_array):
        _rows, _cols = X_np_array.shape
        return da.from_array(X_np_array, chunks=(_rows//5, _cols))


    @staticmethod
    @pytest.fixture
    def X_dask_df(X_pd_df):
        return ddf.from_pandas(X_pd_df, npartitions=5)


    @staticmethod
    @pytest.fixture
    def X_dask_series(X_dask_df):
        return X_dask_df.iloc[:, 0]

    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    y_np_array = np.random.randint(0,10,(100, 1))

    y_pd_df = pd.DataFrame(data=y_np_array, columns=['y'])

    y_pd_series = y_pd_df.iloc[:, 0]

    _rows, _cols = y_np_array.shape
    y_dask_array = da.from_array(y_np_array, chunks=(_rows//5, _cols))

    y_dask_df = ddf.from_pandas(y_pd_df, npartitions=5)

    y_dask_series = y_dask_df.iloc[:, 0]

    # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # pizza fix --- all of these fail except dask_array

    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_np_array(self, X_np_array, y):

        # y cannot be pd.DF, dask.DF for no ravel() attribute
        # y cannot be da.Array because chunks dont match

        if isinstance(y, (pd.core.frame.DataFrame, ddf.core.DataFrame)):
            with pytest.raises(AttributeError):
                dask_LogisticRegression().fit(X_np_array, y)
        elif isinstance(y, da.core.Array):
            with pytest.raises(ValueError):
                dask_LogisticRegression().fit(X_np_array, y)
        else:
            dask_LogisticRegression().fit(X_np_array, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_pd_df(self, X_pd_df, y):

        # NotImplementedError: Could not find signature for add_intercept: <DataFrame>

        with pytest.raises(NotImplementedError):
            dask_LogisticRegression().fit(X_pd_df, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_pd_series(self, X_pd_series, y):
        # NotImplementedError: Could not find signature for add_intercept: <Series>

        with pytest.raises(NotImplementedError):
            dask_LogisticRegression().fit(X_pd_series, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_array(self, X_dask_array, y):

        if isinstance(y, (np.ndarray, pd.core.series.Series, ddf.core.Series)):
            # chunks dont match
            with pytest.raises(ValueError):
                dask_LogisticRegression().fit(X_dask_array, y)
        elif isinstance(y, (pd.core.frame.DataFrame, ddf.core.DataFrame)):
            # no ravel() attribute
            with pytest.raises(AttributeError):
                dask_LogisticRegression().fit(X_dask_array, y)
        else:
            dask_LogisticRegression().fit(X_dask_array, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_df(self, X_dask_df, y):

        # TypeError: This estimator does not support dask dataframes.
        # This might be resolved with one of
        #   1. ddf.to_dask_array(lengths=True)
        #   2. ddf.to_dask_array()  # may cause other issues because of
        #       unknown chunk sizes

        with pytest.raises(TypeError):
            dask_LogisticRegression().fit(X_dask_df, y)


    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series)
    )
    def test_dask_series(self, X_dask_series, y):
        # E       AttributeError: 'Series' object has no attribute 'columns'

        with pytest.raises(AttributeError):
            dask_LogisticRegression().fit(X_dask_series, y)


























