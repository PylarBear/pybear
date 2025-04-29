# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import itertools
from uuid import uuid4

import dask.array as da
import dask.dataframe as ddf
from dask import compute

from pybear.model_selection.GSTCV._GSTCVDask._validation._X_y import _val_X_y



class TestValXy:


    @pytest.mark.parametrize('X',
        (0, True, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('y',
        (0, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_X_y(self, X, y):

        # only accepts dask array, dask series, dask dataFrame

        with pytest.raises(TypeError):
            _val_X_y(X, y)


    def test_rejects_misshapen_y(self, X_da, _rows):

        # multicolumn y
        with pytest.raises(ValueError):
            _val_X_y(
                X_da,
                da.random.randint(0,2,(_rows,2))
            )

        # unequal rows
        with pytest.raises(ValueError):
            _val_X_y(
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


    y_dask_array = da.random.randint(0,2,(_rows, 1)).rechunk((_rows//5, 1))
    y_dask_bad_array = da.random.choice(list('abcde'), (_rows, 1), replace=True)
    y_dask_bad_array = y_dask_bad_array.rechunk((_rows//5, 1))
    y_dask_df = ddf.from_dask_array(y_dask_array, columns=['y'])
    y_dask_bad_df = ddf.from_dask_array(y_dask_bad_array, columns=['y'])
    y_dask_series = y_dask_df.iloc[:, 0]
    y_dask_bad_series = y_dask_bad_df.iloc[:, 0]


    # @pytest.mark.skip(reason=f"pizza keep ur finger on this")
    @pytest.mark.parametrize('X',
        (X_dask_bad_array, X_dask_bad_df, X_dask_bad_series))
    @pytest.mark.parametrize('y',
        (y_dask_array, None)
    )
    def test_rejects_non_numerical_X(self, X, y, _rows, _cols, non_num_X):

        if isinstance(X, da.core.Array):
            # as of 25_04_29 not rejecting in GSTCV, let the estimator do it
            # with pytest.raises(ValueError, match=non_num_X):
            assert _val_X_y(X, y) is None
        elif isinstance(X, (ddf.DataFrame, ddf.Series)):
            # raised by _val_X_y for not da.array
            with pytest.raises(TypeError):
                _val_X_y(X, y)
        else:
            raise Exception


    @pytest.mark.parametrize('X', (X_dask_array, ))
    @pytest.mark.parametrize('y',
        (y_dask_bad_array, y_dask_bad_df, y_dask_bad_series)
    )
    def test_rejects_non_binary_y(self, X, y, non_binary_y):

        if isinstance(X, da.core.Array) \
                and isinstance(y, da.core.Array):
            with pytest.raises(ValueError, match=non_binary_y('GSTCVDask')):
                _val_X_y(X, y)
        elif isinstance(X, (ddf.DataFrame, ddf.Series)) \
                or isinstance(y, (ddf.DataFrame, ddf.Series)):
            # raised by _val_X_y for not da.array
            with pytest.raises(TypeError):
                _val_X_y(X, y)
        else:
            raise Exception


    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X',
        (X_dask_array, X_dask_df, X_dask_series)
    )
    @pytest.mark.parametrize('y',
        (y_dask_array, y_dask_df, y_dask_series, None)
    )
    def test_accuracy(self, X, y, X_COLUMNS):

        if isinstance(X, da.core.Array) \
                and isinstance(y, (da.core.Array, type(None))):
            assert _val_X_y(X, y) is None
        else:
            with pytest.raises(TypeError):
                _val_X_y(X, y)


    # rejects un-chunked Xs and ys
    # @pytest.mark.skip(reason=f"pizza keep ur finger on this")
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
            _val_X_y(unchunked_X_array, unchunked_y_array)
        # END array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



        # df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        unchunked_X_df = ddf.from_dask_array(unchunked_X_array)
        unchunked_y_df = ddf.from_dask_array(unchunked_y_array)

        X_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_X_df.partitions))
        )
        y_row_chunks = compute(
            *itertools.chain(*map(da.shape, unchunked_y_df.partitions))
        )

        # np.nans will return float. just need any of them to be nan.
        # revisit these assertions if ever allowing ddfs into GSTCVDask again
        # assert any(map(isinstance, X_row_chunks, (float for _ in X_row_chunks)))
        # assert any(map(isinstance, y_row_chunks, (float for _ in y_row_chunks)))

        # 25_04_29 blocking dask ddf & series
        if isinstance(unchunked_X_df, (ddf.DataFrame, ddf.Series)) \
                or isinstance(unchunked_y_df, (ddf.DataFrame, ddf.Series)):
            with pytest.raises(TypeError):
                _val_X_y(unchunked_X_df, unchunked_y_df)
            pytest.skip(reason=f"cant do more tests after this fail")

        with pytest.raises(ValueError):
            _val_X_y(unchunked_X_df, unchunked_y_df)
        # END df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *








