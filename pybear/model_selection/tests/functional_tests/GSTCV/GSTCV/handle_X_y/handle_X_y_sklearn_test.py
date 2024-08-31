# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from copy import deepcopy

from uuid import uuid4
import numpy as np
import pandas as pd

from model_selection.GSTCV._GSTCV._handle_X_y._handle_X_y_sklearn import \
    _handle_X_y_sklearn





class TestHandleXySkLearn:


    @pytest.mark.parametrize('X',
        (0, True, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('y',
        (0, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_X_y(self, X, y):

        # only accepts numpy array, pandas series, pandas dataframe,


        with pytest.raises(TypeError):
            _handle_X_y_sklearn(X, y)


    def test_rejects_misshapen_y(self, X_np, _rows):

        # multicolumn y
        with pytest.raises(ValueError):
            _handle_X_y_sklearn(
                X_np,
                np.random.randint(0,2,(_rows,2))
            )


        # unequal rows
        with pytest.raises(ValueError):
            _handle_X_y_sklearn(
                X_np,
                np.random.randint(0,2,(_rows-1,1))
            )


    _rows, _cols = 100, 30

    X_np_array = np.random.randint(0,10,(_rows, _cols))
    X_np_bad_array = \
        np.random.choice(list('abcde'), (_rows, _cols), replace=True)

    X_COLUMNS = [str(uuid4())[:4] for _ in range(_cols)]
    X_pd_df = pd.DataFrame(data=X_np_array, columns=X_COLUMNS)
    X_pd_bad_df = pd.DataFrame(data=X_np_bad_array, columns=X_COLUMNS)
    X_pd_series = pd.Series(X_pd_df.iloc[:, 0], name=X_COLUMNS[0])
    X_pd_bad_series = pd.Series(X_pd_bad_df.iloc[:, 0], name=X_COLUMNS[0])













    y_np_array = np.random.randint(0,2,(_rows, 1))
    y_np_bad_array = np.random.choice(list('abcde'), (_rows, 1), replace=True)
    y_pd_df = pd.DataFrame(data=y_np_array, columns=['y'])
    y_pd_bad_df = pd.DataFrame(data=y_np_bad_array, columns=['y'])
    y_pd_series = pd.Series(y_pd_df.iloc[:, 0], name='y')
    y_pd_bad_series = pd.Series(y_pd_bad_df.iloc[:, 0], name='y')









    @pytest.mark.parametrize('X', (X_np_bad_array, X_pd_bad_df, X_pd_bad_series))
    @pytest.mark.parametrize('y', (y_np_array, None))
    def test_rejects_non_numerical_X(self, X, y, non_num_X):

        _X = deepcopy(X)
        _y = deepcopy(y)

        with pytest.raises(ValueError, match=non_num_X):
            _handle_X_y_sklearn(_X, _y)










    @pytest.mark.parametrize('X', (X_np_array, ))
    @pytest.mark.parametrize('y', (y_np_bad_array, y_pd_bad_df, y_pd_bad_series))
    def test_rejects_non_binary_y(self, X, y, non_binary_y):

        _X = deepcopy(X)
        _y = deepcopy(y)

        with pytest.raises(ValueError, match=non_binary_y('GSTCV')):
            _handle_X_y_sklearn(_X, _y)




    @pytest.mark.parametrize('_rows', (_rows,))
    @pytest.mark.parametrize('_cols', (_cols,))
    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X', (X_np_array, X_pd_df, X_pd_series))
    @pytest.mark.parametrize('y', (y_np_array, y_pd_df, y_pd_series, None))
    def test_accuracy(self, X, y, _rows, _cols, X_COLUMNS):

        _X = deepcopy(X)
        _y = deepcopy(y)

        out_X, out_y, out_feature_names, out_n_features = \
            _handle_X_y_sklearn(_X, _y)

        assert isinstance(out_X, np.ndarray)
        if 'series' in str(type(_X)).lower():
            assert out_X.shape == (_rows, 1)
            assert out_n_features == 1
            assert np.array_equiv(out_X, _X.to_numpy().reshape((_rows, 1)))
            assert out_X.dtype == _X.dtype
            assert np.array_equiv(out_feature_names, X_COLUMNS[:1])
        elif 'dataframe' in str(type(_X)).lower():
            assert out_X.shape == (_rows, _cols)
            assert out_n_features == _cols
            assert np.array_equiv(out_X, _X.to_numpy())
            assert len(np.unique(_X.dtypes)) == 1
            assert out_X.dtype == np.unique(_X.dtypes)[0]
            assert np.array_equiv(out_feature_names, X_COLUMNS)
        else:
            assert out_X.shape == (_rows, _cols)
            assert out_n_features == _cols
            assert np.array_equiv(out_X, _X)
            assert out_X.dtype == _X.dtype
            assert out_feature_names is None


        if _y is None:
            assert isinstance(out_y, type(None))
        else:
            assert isinstance(out_y, np.ndarray)
            if 'series' in str(type(_y)).lower():
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.to_numpy().ravel())
                assert out_y.dtype == _y.dtype
            elif 'dataframe' in str(type(_y)).lower():
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.to_numpy().ravel())
                assert len(np.unique(_y.dtypes)) == 1
                assert out_y.dtype == np.unique(_y.dtypes)[0]
            else:
                assert out_y.shape == (_rows,)
                assert np.array_equiv(out_y, _y.ravel())
                assert out_y.dtype == _y.dtype






































