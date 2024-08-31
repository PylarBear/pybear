# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import inspect

import dask.array as da

from dask_ml.model_selection import KFold as dask_KFold


# KFold().split(self, X, y=None, groups=None)
# dask_KFold: y is not required and when passed is not used.

pytest.skip(reason=f'for edification purposes only', allow_module_level=True)
# no need to test the volatile and unpredictable state of dask error messages


class TestDaskKFold:

    # AS OF 24_02_24_17_05_00 ONLY DASK ARRAYS CAN BE PASSED TO
    # dask_KFOLD (NOT np, pd.DF, dask.DF)
    # VERIFIED AGAIN 24_06_27_09_08_00

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def X_dask_series(X_ddf):
        return X_ddf.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def X_pd_series(X_pd):
        return X_pd.iloc[:, 0]


    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # all of these fail except dask_array


    def test_np_array(self, X_np, _client):

        with pytest.raises(AttributeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_np, y=None)
            assert inspect.isgenerator(out)

            for train_idxs, test_idxs in out:
                # accessing the generator is when the error happens
                pass


    def test_pd_df(self, X_pd, _client):

        with pytest.raises(AttributeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_pd, y=None)
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


    def test_dask_array(self, X_da, _client):
        _n_splits = 10
        out = dask_KFold(n_splits=_n_splits).split(X_da, y=None)

        for train_idxs, test_idxs in out:

            assert inspect.isgenerator(out)
            assert isinstance(train_idxs, da.core.Array)
            assert len(train_idxs.shape) == 1
            assert len(train_idxs) == \
                        int(X_da.shape[0] * (_n_splits-1) / _n_splits)

            assert isinstance(test_idxs, da.core.Array)
            assert len(test_idxs.shape) == 1
            assert len(test_idxs) == \
                        int(X_da.shape[0] * 1 / _n_splits)


    def test_dask_df(self, X_ddf, _client):

        # TypeError: This estimator does not support dask dataframes.

        with pytest.raises(TypeError):
            _n_splits = 10
            out = dask_KFold(n_splits=_n_splits).split(X_ddf, y=None)
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

























