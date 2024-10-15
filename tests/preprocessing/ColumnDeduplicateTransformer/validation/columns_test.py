# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation._columns \
    import _val_columns

import numpy as np
import pandas as pd
from copy import deepcopy


import pytest



class TestColumns:

    # fixtures ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 5)

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_X_factory, _shape):
        return _X_factory(_format='np', _shape=_shape)

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_pd(_X_np, _columns):
        return pd.DataFrame(
        data=_X_np,
        columns=_columns
    )
    # END fixtures ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_columns',
        (-1, 0, 1, 3.14, True, 'trash', {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, _X_np, _X_pd, junk_columns):
        with pytest.raises(TypeError):
            _val_columns(junk_columns, _X_np)

        with pytest.raises(TypeError):
            _val_columns(junk_columns, _X_pd)


    @pytest.mark.parametrize('bad_columns',
        ([0,1,2,3,4], np.random.uniform(0,1,(5,)))
    )
    def test_rejects_bad_type(self, _X_np, _X_pd, bad_columns):
        with pytest.raises(ValueError):
            _val_columns(bad_columns, _X_np)

        with pytest.raises(ValueError):
            _val_columns(bad_columns, _X_pd)


    @pytest.mark.parametrize('bad_columns',
        ([0,1], [True, False], np.random.uniform(0,1,(3,)))
    )
    def test_rejects_bad_len(self, _X_np, _X_pd, bad_columns):
        with pytest.raises(ValueError):
            _val_columns(bad_columns, _X_np)

        with pytest.raises(ValueError):
            _val_columns(bad_columns, _X_pd)


    @pytest.mark.parametrize('_type',
        ('list', 'tuple', 'set', 'ndarray_1d', 'ndarray_2d','None')
    )
    def test_accepts_good(self, _X_np, _X_pd, _columns, _type):

        if _type == 'list':
            good_columns = deepcopy(_columns)
        elif _type =='tuple':
            good_columns = tuple(deepcopy(_columns))
        elif _type == 'set':
            good_columns = set(deepcopy(_columns))
        elif _type == 'ndarray_1d':
            good_columns = np.array(deepcopy(_columns))
        elif _type == 'ndarray_2d':
            good_columns = np.array(deepcopy(_columns)).reshape((1,-1))
        elif _type == 'None':
            good_columns = None

        _val_columns(good_columns, _X_np)

        _val_columns(good_columns, _X_pd)








