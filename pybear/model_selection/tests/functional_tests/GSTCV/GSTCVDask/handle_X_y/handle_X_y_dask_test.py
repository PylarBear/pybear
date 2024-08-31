# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from copy import deepcopy
import itertools
from uuid import uuid4
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
from dask import compute

from model_selection.GSTCV._GSTCVDask._handle_X_y._handle_X_y_dask import \
    _handle_X_y_dask


class TestHandleXyDask:


    @pytest.mark.parametrize('X',
        (0, True, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('y',
        (0, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_X_y(self, X, y):

        # only accepts numpy array, pandas series, pandas dataframe,
        # dask array, dask series, dask dataFrame

        with pytest.raises(TypeError):
            _handle_X_y_dask(X, y)


    def test_rejects_misshapen_y(self, X_da, _rows):

        # multicolumn y
        with pytest.raises(ValueError):
            _handle_X_y_dask(
                X_da,
                da.random.randint(0,2,(_rows,2))
            )


        # unequal rows
        with pytest.raises(ValueError):
            _handle_X_y_dask(
                X_da,
                da.random.randint(0,2,(_rows-1,1))
            )


    _rows, _cols = 100, 30

    X_dask_array = \
        da.random.randint(0,10,(_rows, _cols)).rechunk((_rows//5, _cols))
    X_dask_bad_array = \
        da.random.choice(list('abcde'), (_rows, _cols), replace=True)
    X_dask_bad_array = X_dask_bad_array.rechunk((_rows//5, _cols))
    X_COLUMNS = [str(uuid4())[:4] for _ in range(_cols)]
    X_dask_df = ddf.from_dask_array(X_dask_array, columns=X_COLUMNS)
    X_dask_bad_df = ddf.from_dask_array(X_dask_bad_array, columns=X_COLUMNS)
    X_dask_series = X_dask_df.iloc[:, 0]
    X_dask_bad_series = X_dask_bad_df.iloc[:, 0]
    X_np_array = X_dask_array.compute()
    X_np_bad_array = X_dask_bad_array.compute()
    X_pd_df = X_dask_df.compute()
    X_pd_bad_df = X_dask_bad_df.compute()
    X_pd_series = pd.Series(X_pd_df.iloc[:, 0], name=X_COLUMNS[0])
    X_pd_bad_series = pd.Series(X_pd_bad_df.iloc[:, 0], name=X_COLUMNS[0])


    y_dask_array = da.random.randint(0,2,(_rows, 1)).rechunk((_rows//5, 1))
    y_dask_bad_array = da.random.choice(list('abcde'), (_rows, 1), replace=True)
    y_dask_bad_array = y_dask_bad_array.rechunk((_rows//5, 1))
    y_dask_df = ddf.from_dask_array(y_dask_array, columns=['y'])
    y_dask_bad_df = ddf.from_dask_array(y_dask_bad_array, columns=['y'])
    y_dask_series = y_dask_df.iloc[:, 0]
    y_dask_bad_series = y_dask_bad_df.iloc[:, 0]
    y_np_array = y_dask_array.compute()
    y_np_bad_array = y_dask_bad_array.compute()
    y_pd_df = y_dask_df.compute()
    y_pd_bad_df = y_dask_bad_df.compute()
    y_pd_series =y_pd_df.iloc[:, 0]
    y_pd_bad_series = y_pd_bad_df.iloc[:, 0]



    @pytest.mark.parametrize('X',
        (X_np_bad_array, X_pd_bad_df, X_pd_bad_series, X_dask_bad_array,
         X_dask_bad_df, X_dask_bad_series))
    @pytest.mark.parametrize('y',
        (y_np_array, None)
    )
    def test_rejects_non_numerical_X(self, X, y, _rows, _cols, non_num_X):

        _X = deepcopy(X)
        _y = deepcopy(y)

        with pytest.raises(ValueError, match=non_num_X):
            _handle_X_y_dask(_X, _y)


    @pytest.mark.parametrize('X', (X_dask_array, ))
    @pytest.mark.parametrize('y',
        (y_dask_bad_array, y_dask_bad_df, y_dask_bad_series,
         y_np_bad_array, y_pd_bad_df, y_pd_bad_series)
    )
    def test_rejects_non_binary_y(self, X, y, non_binary_y):

        _X = deepcopy(X)
        _y = deepcopy(y)

        with pytest.raises(ValueError, match=non_binary_y('GSTCVDask')):
            _handle_X_y_dask(_X, _y)



    @pytest.mark.parametrize('_rows', (_rows,))
    @pytest.mark.parametrize('_cols', (_cols,))
    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X',
        (X_np_array, X_pd_df, X_pd_series, X_dask_array, X_dask_df, X_dask_series)
    )
    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df,
         y_dask_series, None)
    )
    def test_accuracy(self, X, y, _rows, _cols, X_COLUMNS):

        _X = deepcopy(X)
        _y = deepcopy(y)

        out_X, out_y, out_feature_names, out_n_features = \
            _handle_X_y_dask(_X, _y)

        # for convenience, compute _X and _y to test against out
        try:
            _X = _X.compute()
        except:
            pass

        try:
            _y = _y.compute()
        except:
            pass


        assert isinstance(out_X, da.core.Array)
        if 'series' in str(type(_X)).lower():
            assert out_X.shape == (_rows, 1)
            assert out_n_features == 1
            assert np.array_equiv(out_X, _X.to_numpy().reshape((_rows, 1)))
            # assert out_X.dtype == _X.dtype
            assert np.array_equiv(out_feature_names, X_COLUMNS[:1])
        elif 'dataframe' in str(type(_X)).lower():
            assert out_X.shape == (_rows, _cols)
            assert out_n_features == _cols
            assert np.array_equiv(out_X, _X.to_numpy())
            assert len(np.unique(list(map(str, _X.dtypes)))) == 1
            # assert out_X.dtype == np.unique(list(map(str, _X.dtypes)))[0]
            assert np.array_equiv(out_feature_names, X_COLUMNS)
        else:
            assert out_X.shape == (_rows, _cols)
            assert out_n_features == _cols
            assert np.array_equiv(out_X, _X)
            # assert out_X.dtype == _X.dtype
            assert out_feature_names is None


        if _y is None:
            assert isinstance(out_y, type(None))
        else:
            assert isinstance(out_y, da.core.Array)
            if 'series' in str(type(_y)).lower():
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.to_numpy().ravel())
                # assert out_y.dtype == _y.dtype
            elif 'dataframe' in str(type(_y)).lower():
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.to_numpy().ravel())
                assert len(np.unique(_y.dtypes)) == 1
                # assert out_y.dtype == np.unique(_y.dtypes)[0]
            else:
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.ravel())
                # assert out_y.dtype == _y.dtype



    # rejects un-chunked Xs and ys
    @pytest.mark.parametrize(f'X_dask_df, y_dask_df', ((X_dask_df, y_dask_df),))
    def test_unchunked(self, X_dask_df, y_dask_df):

        # array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        unchunked_X_array = X_dask_df.to_dask_array()   # no 'lengths'!
        unchunked_y_array = y_dask_df.to_dask_array()  # no 'lengths'!

        X_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_X_array.blocks))
        )
        y_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_y_array.blocks))
        )

        # np.nans will return float. just need any of them to be nan.
        assert any(map(isinstance, X_row_chunks, (float for _ in X_row_chunks)))
        assert any(map(isinstance, y_row_chunks, (float for _ in y_row_chunks)))

        with pytest.raises(ValueError):
            _handle_X_y_dask(unchunked_X_array, unchunked_y_array)
        # END array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



        # df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        unchunked_X_df = da.array(unchunked_X_array)
        unchunked_y_df = da.array(unchunked_y_array)

        X_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_X_df.partitions))
        )
        y_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_y_df.partitions))
        )

        # np.nans will return float. just need any of them to be nan.
        assert any(map(isinstance, X_row_chunks, (float for _ in X_row_chunks)))
        assert any(map(isinstance, y_row_chunks, (float for _ in y_row_chunks)))

        with pytest.raises(ValueError):
            _handle_X_y_dask(unchunked_X_array, unchunked_y_array)
        # END df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *






















