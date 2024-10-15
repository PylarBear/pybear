# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _column_getter import _column_getter

from pybear.utilities._nan_masking import nan_mask

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest


@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
class TestColumnGetter:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 3)

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_num(_X_factory, _shape, _has_nan):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_str(_X_factory, _shape, _has_nan):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='str',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_type',
        (
        'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_col_idx1', (0, 1, 2))
    @pytest.mark.parametrize('_col_idx2', (0, 1, 2))
    def test_accuracy(
        self, _has_nan, _dtype, _type, _col_idx1, _col_idx2, _shape, _X_num,
        _X_str, _master_columns
    ):

        if _dtype == 'str' and _type not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str

        if _type == 'ndarray':
            _X_wip = _X
        elif _type == 'df':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_master_columns.copy()[:_shape[1]]
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
            _X_wip = ss._bsr.bsr_array(_X)
        else:
            raise Exception

        column1, column2 = _column_getter(_X_wip, _col_idx1, _col_idx2)

        assert len(column1.shape) == 1
        assert len(column2.shape) == 1

        if _dtype == 'num':
            assert np.array_equal(column1, _X[:, _col_idx1], equal_nan=True)
            assert np.array_equal(column2, _X[:, _col_idx2], equal_nan=True)
        elif _dtype == 'str':
            # since changed column_getter to assign np.nan to nan-likes,
            # need to accommodate these np.nans when doing array_equal.
            # as of 24_10_15, array_equal equal_nan cannot cast for str types
            if _has_nan:
                NOT_NAN1 = np.logical_not(nan_mask(column1)).astype(bool)
                assert np.array_equal(
                    column1[NOT_NAN1], _X[:, _col_idx1][NOT_NAN1]
                )
                NOT_NAN2 = np.logical_not(nan_mask(column2)).astype(bool)
                assert np.array_equal(
                    column2[NOT_NAN2], _X[:, _col_idx2][NOT_NAN2]
                )
            else:
                assert np.array_equal(column1, _X[:, _col_idx1])
                assert np.array_equal(column2, _X[:, _col_idx2])




