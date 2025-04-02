# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text.__shared._transform._map_X_to_list import \
    _map_X_to_list

import pytest

import numpy as np
import pandas as pd
import polars as pl



class TestMapXToList:

    # def _map_X_to_list(
    #     _X: Union[Dim1Types, Dim2Types]
    # ) -> Union[list[str], list[list[str]]]:



    @pytest.mark.parametrize('_container',
        ('py_list', 'py_tuple', 'py_set', 'np', 'pd', 'pl')
    )
    def test_accuracy_1D(self, _container):

        _base_X = np.random.choice(list('abcdefghi'), 10, replace=True)

        if _container == 'py_list':
            _X = _base_X.tolist()
        elif _container == 'py_tuple':
            _X = tuple(_base_X.tolist())
        elif _container == 'py_set':
            _X = set(_base_X.tolist())
        elif _container == 'np':
            _X = _base_X.copy()
        elif _container == 'pd':
            _X = pd.Series(_base_X.copy())
        elif _container == 'pl':
            _X = pl.Series(_base_X.copy())
        else:
            raise Exception


        out = _map_X_to_list(_X)

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        if _container == 'py_set':
            assert np.array_equal(sorted(out), sorted(list(_X)))
        else:
            assert np.array_equal(out, _base_X)



    @pytest.mark.parametrize('_container',
        ('py_list', 'py_tuple', 'np', 'pd', 'pl')
    )
    def test_accuracy_2D(self, _container):

        _base_X = np.random.choice(list('abcdefghi'), (37, 13), replace=True)

        if _container == 'py_list':
            _X = list(map(list, _base_X))
        elif _container == 'py_tuple':
            _X = tuple(map(tuple, _base_X))
        elif _container == 'np':
            _X = _base_X.copy()
        elif _container == 'pd':
            _X = pd.DataFrame(_base_X.copy())
        elif _container == 'pl':
            _X = pl.DataFrame(_base_X.copy())
        else:
            raise Exception


        out = _map_X_to_list(_X)

        assert isinstance(out, list)
        for idx, row in enumerate(out):
            assert isinstance(row, list)
            assert all(map(isinstance, row, (str for _ in row)))
            assert np.array_equal(row, _base_X[idx])






