# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates

import numpy as np
import pandas as pd
import scipy.sparse as ss
from uuid import uuid4

import pytest




class Fixtures:

    @staticmethod
    @pytest.fixture()
    def _cols():
        return 20


    @staticmethod
    @pytest.fixture()
    def _dupl1():
        return [
            [0, 17],
            [1, 8, 11]
        ]


    @staticmethod
    @pytest.fixture()
    def _dupl2():
        return []


    @staticmethod
    @pytest.fixture()
    def _dupl3():
        return [
            [0, 7, 9],
            [6, 8, 13, 15]
        ]


    @staticmethod
    @pytest.fixture()
    def _X1(_cols, _dupl1):
        _ = np.random.randint(0, 3, (100, _cols))
        for _set in _dupl1:
            for _col in _set[1:]:
                _[:, _col] = _[:, _set[0]]

        return _


    @staticmethod
    @pytest.fixture()
    def _X2(_cols, _dupl2):
        _ = np.random.randint(0, 3, (100, _cols))
        for _set in _dupl2:
            for _col in _set[1:]:
                _[:, _col] = _[:, _set[0]]

        return _


    @staticmethod
    @pytest.fixture()
    def _X3(_cols, _dupl3):
        _ = np.random.randint(0, 3, (100, _cols))
        for _set in _dupl3:
            for _col in _set[1:]:
                _[:, _col] = _[:, _set[0]]

        return _


class TestFindDuplicates(Fixtures):

    @pytest.mark.parametrize('_n_jobs', (-1, 1,2,3,4))
    @pytest.mark.parametrize('_trial', (1, 2, 3))
    @pytest.mark.parametrize('_type',
    (
     'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
     'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
     'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
    )
    )
    def test_accuracy(
        self, _n_jobs, _trial, _type, _X1, _X2, _X3, _dupl1, _dupl2, _dupl3, _cols
    ):

        if _trial == 1:
            _X = _X1
        elif _trial == 2:
            _X = _X2
        elif _trial == 3:
            _X = _X3

        if _type == 'ndarray':
            _X_wip = _X
        elif _type == 'df':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=[str(uuid4())[:4] for _ in range(_cols)]
            )
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

        out = _find_duplicates(_X_wip, _n_jobs)

        for _idx in range(len(out)):
            if _trial == 1:
                assert np.array_equal(out[_idx], _dupl1[_idx])
            elif _trial == 2:
                assert out == []
                assert np.array_equal(out[_idx], _dupl2[_idx])
            elif _trial == 3:
                assert np.array_equal(out[_idx], _dupl3[_idx])

























