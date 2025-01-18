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
    @pytest.mark.parametrize('_format',
        (
        'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_col_idx1', (0, 1, 2))
    def test_accuracy(
        self, _has_nan, _dtype, _format, _col_idx1, _shape, _X_num, _X_str,
        _master_columns
    ):

        # coo, dia, & bsr matrix/array are blocked. should raise here.

        if _dtype == 'str' and _format not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str

        if _format == 'ndarray':
            _X_wip = _X
        elif _format == 'df':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_master_columns.copy()[:_shape[1]]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        else:
            raise Exception

        if isinstance(_X_wip, (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
            ss.dia_array, ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _column_getter(_X_wip, _col_idx1)
            pytest.skip(reason=f"cant do more tests after exception")
        else:
            column1 = _column_getter(_X_wip, _col_idx1)

        assert len(column1.shape) == 1

        # if running scipy sparse, then column1 will be hstack((indices, values)).
        # take it easy on yourself, just transform this output to a regular
        # np array to ensure the correct column is being pulled
        if _format not in ('ndarray', 'df'):
            new_column1 = np.zeros(_shape[0]).astype(np.float64)
            new_column1[column1[:len(column1)//2].astype(np.int32)] = \
                column1[len(column1)//2:]
            column1 = new_column1
            del new_column1


        if _dtype == 'num':
            assert np.array_equal(column1, _X[:, _col_idx1], equal_nan=True)
        elif _dtype == 'str':
            assert np.array_equal(
                column1.astype(str),
                _X[:, _col_idx1].astype(str)
            )























