# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import scipy.sparse as ss

from pybear.preprocessing._InterceptManager._partial_fit._column_getter import \
    _column_getter

from pybear.utilities._nan_masking import nan_mask_string



class TestColumnGetter:


    @pytest.mark.parametrize('_format',
        (
             'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
             'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
             'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_blocks_coo_dia_bsr(self, _X_factory, _format, _shape):

        # _columns_getter only allows ss that are indexable

        _X_num = _X_factory(_has_nan=False, _shape=_shape)

        if _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X_num)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X_num)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X_num)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X_num)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X_num)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X_num)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X_num)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X_num)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X_num)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X_num)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X_num)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X_num)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X_num)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X_num)
        else:
            raise Exception

        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
             ss.dia_array, ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _column_getter(_X_wip, 0)
        else:
            _column_getter(_X_wip, 0)


    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_format',
        (
        'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'lil_matrix',
        'dok_matrix', 'csr_array', 'csc_array', 'lil_array', 'dok_array'
        )
    )
    @pytest.mark.parametrize('_col_idx1', (0, 1, 2))
    def test_accuracy(
        self, _X_factory, _dtype, _format, _col_idx1, _shape, _columns
    ):

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        if _dtype == 'str' and _format not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if _dtype == 'num':
            _X = _X_factory(_has_nan=False, _shape=_shape)

        elif _dtype == 'str':
            _X = _X_factory(
                _format='np', _dtype='str', _has_nan=False, _shape=_shape
            )

        if _format == 'ndarray':
            _X_wip = _X
        elif _format == 'df':
            _X_wip = pd.DataFrame(data=_X, columns=_columns)
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

        column1 = _column_getter(_X_wip, _col_idx1)
        assert isinstance(column1, np.ndarray)
        assert len(column1.shape) == 1


        # since all the various _X_wips came from _X, just use _X to referee
        # whether _column_getter pulled the correct column from _X_wip
        assert np.array_equal(column1, _X[:, _col_idx1])




