# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from uuid import uuid4

import numpy as np
import pandas as pd

from pybear.model_selection.GSTCV._GSTCV._validation._X_y import _val_X_y





class TestValXy:


    # @pytest.mark.skip(reason=f"pizza keep ur finger on this")
    @pytest.mark.parametrize('X',
        (0, True, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('y',
        (0, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_X_y(self, X, y):

        with pytest.raises(TypeError):
            _val_X_y(X, y)




    def test_rejects_misshapen_y(self, X_np, _rows):

        # multicolumn y
        with pytest.raises(ValueError):
            _val_X_y(
                X_np,
                np.random.randint(0,2,(_rows,2))
            )


        # unequal rows
        with pytest.raises(ValueError):
            _val_X_y(
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






    # @pytest.mark.skip(reason=f"pizza keep ur finger on this")
    @pytest.mark.parametrize('X', (X_np_bad_array, X_pd_bad_df, X_pd_bad_series))
    @pytest.mark.parametrize('y', (y_np_array, ))
    def test_rejects_non_numerical_X(self, X, y, non_num_X):

        # with pytest.raises(ValueError, match=non_num_X):
        assert _val_X_y(X, y) is None




    @pytest.mark.parametrize('X', (X_np_array, ))
    @pytest.mark.parametrize('y', (y_np_bad_array, y_pd_bad_df, y_pd_bad_series))
    def test_rejects_non_binary_y(self, X, y, non_binary_y):

        with pytest.raises(ValueError, match=non_binary_y('GSTCV')):
            _val_X_y(X, y)







    @pytest.mark.parametrize('X_COLUMNS', (X_COLUMNS,))
    @pytest.mark.parametrize('X', (X_np_array, X_pd_df, X_pd_series))
    @pytest.mark.parametrize('y', (y_np_array, y_pd_df, y_pd_series))
    def test_accuracy(self, X, y, X_COLUMNS):

        assert _val_X_y(X, y) is None





