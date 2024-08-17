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
import distributed

from dask_ml.model_selection import KFold as dask_KFold


# KFold().split(self, X, y=None, groups=None)
# dask_KFold: y is not required and when passed is not used.


class TestDaskKFold:

    # AS OF 24_02_24_17_05_00 ONLY DASK ARRAYS CAN BE PASSED TO
    # dask_KFOLD (NOT np, pd.DF, dask.DF)
    # VERIFIED AGAIN 24_06_27_09_08_00

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @staticmethod
    @pytest.fixture
    def X_dask_array():
        return da.random.randint(0, 10, (100, 30)).rechunk((20, 30))


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


    @staticmethod
    @pytest.fixture(scope='module')
    def _client():
        client = distributed.Client(n_workers=1, threads_per_worker=1)
        yield client
        client.close()

    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # all of these fail except dask_array


    def test_np_array(self, X_np_array, _client):

        with pytest.raises(AttributeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_np_array, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass


    def test_pd_df(self, X_pd_df, _client):

        with pytest.raises(AttributeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_pd_df, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass


    def test_pd_series(self, X_pd_series, _client):

        # ValueError: Expected a 2-dimensional container but got <class
        # 'pandas.core.series.Series'> instead. Pass a DataFrame containing
        # a single row (i.e. single sample) or a single column (i.e. single
        # feature) instead.

        with pytest.raises(ValueError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_pd_series, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass


    def test_dask_array(self, X_dask_array, _client):
        _n_splits = 10
        out = dask_KFold(n_splits=_n_splits).split(X_dask_array, y=None)

        for train_idxs, test_idxs in out:

            assert inspect.isgenerator(out)
            assert isinstance(train_idxs, da.core.Array)
            assert len(train_idxs.shape) == 1
            assert len(train_idxs) == \
                        int(X_dask_array.shape[0] * (_n_splits-1) / _n_splits)

            assert isinstance(test_idxs, da.core.Array)
            assert len(test_idxs.shape) == 1
            assert len(test_idxs) == \
                        int(X_dask_array.shape[0] * 1 / _n_splits)


    def test_dask_df(self, X_dask_df, _client):

        # TypeError: This estimator does not support dask dataframes.

        with pytest.raises(TypeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_dask_df, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass


    def test_dask_series(self, X_dask_series, _client):

        # ValueError: Expected a 2-dimensional container but got <class
        # 'pandas.core.series.Series'> instead. Pass a DataFrame containing
        # a single row (i.e. single sample) or a single column (i.e. single
        # feature) instead.

        with pytest.raises(ValueError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_dask_series, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass

























