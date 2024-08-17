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


    def test_rejects_misshapen_y(self):

        # multicolumn y
        with pytest.raises(ValueError):
            _handle_X_y_dask(
                da.random.randint(0,10,(100,30)),
                da.random.randint(0,10,(100,2))
            )


        # unequal rows
        with pytest.raises(ValueError):
            _handle_X_y_dask(
                da.random.randint(0,10,(100,30)),
                da.random.randint(0,10,(99,1))
            )








    _rows, _cols = 100, 30

    X_dask_array = da.random.randint(0,10,(_rows, _cols)).rechunk((_rows//5, _cols))
    X_COLUMNS = [str(uuid4())[:4] for _ in range(X_dask_array.shape[1])]
    X_dask_df = ddf.from_dask_array(X_dask_array, columns=X_COLUMNS)
    X_dask_series = X_dask_df.iloc[:, 0]
    X_np_array = X_dask_array.compute()
    X_pd_df = X_dask_df.compute()
    X_pd_series = pd.Series(X_pd_df.iloc[:, 0], name=X_COLUMNS[0])

    y_dask_array = da.random.randint(0,10,(_rows, 1)).rechunk((_rows//5, _cols))
    y_dask_df = ddf.from_dask_array(y_dask_array, columns=['y'])
    y_dask_series = y_dask_df.iloc[:, 0]
    y_np_array = y_dask_array.compute()
    y_pd_df = y_dask_df.compute()
    y_pd_series =y_pd_df.iloc[:, 0]




    @pytest.mark.parametrize('_rows', (_rows,))
    @pytest.mark.parametrize('_cols', (_cols,))
    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X',
        (X_np_array, X_pd_df, X_pd_series, X_dask_array, X_dask_df, X_dask_series)
    )
    @pytest.mark.parametrize('y',
        (y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series, None)
    )
    def test_accuracy(self, X, y, _rows, _cols, X_COLUMNS):

        _X = deepcopy(X)
        _y = deepcopy(y)

        out_X, out_y, out_feature_names, out_n_features = _handle_X_y_dask(_X, _y)

        assert isinstance(out_X, da.core.Array)
        if 'series' in str(type(X)).lower():
            assert out_X.shape == (_rows, 1)
        else:
            assert out_X.shape == (_rows, _cols)

        if y is None:
            assert isinstance(out_y, type(None))
        else:
            assert isinstance(out_y, da.core.Array)
            assert out_y.shape == (_rows,)


        if 'DataFrame' in str(type(X)):
            assert np.array_equiv(out_feature_names, X_COLUMNS)
        elif 'Series' in str(type(X)):
            assert np.array_equiv(out_feature_names, X_COLUMNS[:1])
        else:
            assert out_feature_names is None


        if 'series' in str(type(X)).lower():
            assert out_n_features == 1
        else:
            assert out_n_features == _cols




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






















