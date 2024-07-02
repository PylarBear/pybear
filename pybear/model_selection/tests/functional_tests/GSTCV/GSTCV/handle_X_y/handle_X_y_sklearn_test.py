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


    def test_rejects_misshapen_y(self):

        # multicolumn y
        with pytest.raises(ValueError):
            _handle_X_y_sklearn(
                np.random.randint(0,10,(100,30)),
                np.random.randint(0,10,(100,2))
            )


        # unequal rows
        with pytest.raises(ValueError):
            _handle_X_y_sklearn(
                np.random.randint(0,10,(100,30)),
                np.random.randint(0,10,(99,1))
            )








    _rows, _cols = 100, 30

    X_np_array = np.random.randint(0,10,(_rows, _cols))
    X_COLUMNS = [str(uuid4())[:4] for _ in range(X_np_array.shape[1])]
    X_pd_df = pd.DataFrame(data=X_np_array, columns=X_COLUMNS)
    X_pd_series = pd.Series(X_pd_df.iloc[:, 0], name=X_COLUMNS[0])


    y_np_array = np.random.randint(0,10,(_rows, 1))
    y_pd_df = pd.DataFrame(data=y_np_array, columns=['y'])
    y_pd_series = pd.Series(y_pd_df.iloc[:, 0], name='y')



    @pytest.mark.parametrize('_rows', (_rows,))
    @pytest.mark.parametrize('_cols', (_cols,))
    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X', (X_np_array, X_pd_df, X_pd_series))
    @pytest.mark.parametrize('y', (y_np_array, y_pd_df, y_pd_series, None))
    def test_accuracy(self, X, y, _rows, _cols, X_COLUMNS):

        _X = deepcopy(X)
        _y = deepcopy(y)

        out_X, out_y, out_feature_names, out_n_features = _handle_X_y_sklearn(_X, _y)

        assert isinstance(out_X, np.ndarray)
        if 'series' in str(type(_X)).lower():
            assert out_X.shape == (_rows, 1)
        else:
            assert out_X.shape == (_rows, _cols)

        if _y is None:
            assert isinstance(out_y, type(None))
        else:
            assert isinstance(out_y, np.ndarray)
            assert out_y.shape == (_rows,)


        if 'DataFrame' in str(type(_X)):
            assert np.array_equiv(out_feature_names, X_COLUMNS)
        elif 'Series' in str(type(_X)):
            assert np.array_equiv(out_feature_names, X_COLUMNS[:1])
        else:
            assert out_feature_names is None


        if 'series' in str(type(_X)).lower():
            assert out_n_features == 1
        else:
            assert out_n_features == _cols
































