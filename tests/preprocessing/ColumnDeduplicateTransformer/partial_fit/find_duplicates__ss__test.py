# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates

import numpy as np
import scipy.sparse as ss

import pytest




class TestNpFindDuplicates:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl1():
        return [
            [0, 7],
            [1, 5, 8]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl2():
        return []


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl3():
        return [
            [0, 7, 9],
            [1, 5, 6, 8]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl4():
        return [[0, 4, 7]]


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_dupl_set', (1, 2, 3, 4))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_type',
    (
     'ndarray', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
     'lil_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
     'dia_array', 'lil_array', 'dok_array'
    )
    )
    def test_accuracy(
        self, _X_factory, _dtype, _equal_nan, _type, _shape, _has_nan,
        _dupl_set, _dupl1, _dupl2, _dupl3, _dupl4
    ):


        if _dupl_set == 1:
            _dupl = _dupl1
        elif _dupl_set == 2:
            _dupl = _dupl2
        elif _dupl_set == 3:
            _dupl = _dupl3
        elif _dupl_set == 4:
            _dupl = _dupl4
        else:
            raise Exception

        _X = _X_factory(
            _dupl=_dupl,
            _format='np',
            _dtype=_dtype,
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.25,
            _shape=_shape
        )


        if _type == 'ndarray':
            _X_wip = _X
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
        else:
            raise Exception

        out = _find_duplicates(
            _X_wip, _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan, _n_jobs=-1
        )

        if (not _equal_nan and _has_nan) or _dupl_set in [2]:
            assert out == []
        elif _dupl_set == 1:
            assert len(out) == len(_dupl1)
            for _idx in range(len(out)):
                assert np.array_equal(out[_idx], _dupl1[_idx])
        elif _dupl_set == 3:
            assert len(out) == len(_dupl3)
            for _idx in range(len(out)):
                assert np.array_equal(out[_idx], _dupl3[_idx])
        elif _dupl_set == 4:
            assert len(out) == len(_dupl4)
            for _idx in range(len(out)):
                assert np.array_equal(out[_idx], _dupl4[_idx])
        else:
            raise Exception

























