# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _column_getter import _column_getter

import numpy as np
import pandas as pd
import scipy.sparse as ss
from uuid import uuid4

import pytest



class Fixtures:

    @staticmethod
    @pytest.fixture()
    def _cols():
        return 3

    @staticmethod
    @pytest.fixture()
    def _X(_cols):
        return np.random.randint(0, 3, (100, _cols))



class TestColumnComparer(Fixtures):


    @pytest.mark.parametrize('_type',
        (
        'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_col_idx1', (0, 1, 2))
    @pytest.mark.parametrize('_col_idx2', (0, 1, 2))
    def test_accuracy(self, _type, _col_idx1, _col_idx2, _X, _cols):

        if _type == 'ndarray':
            _X_wip = _X
        elif _type == 'df':
            _X_wip = pd.DataFrame(data=_X, columns=[str(uuid4())[:4] for _ in _X.T])
        elif _type == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _type == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _type == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _type == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _type == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _type == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _type == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
        elif _type == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _type == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _type == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _type == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _type == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _type == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _type == 'bsr_array':
            _X_wip = ss._bsr.bsr_matrix(_X)

        column1, column2 = _column_getter(_X_wip, _col_idx1, _col_idx2)

        assert np.array_equal(column1, _X[:, _col_idx1])

        assert np.array_equal(column2, _X[:, _col_idx2])





