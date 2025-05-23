# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _column_getter import _column_getter



class TestColumnGetter:


    @pytest.mark.parametrize('_dtype', ('flt', 'str'))
    @pytest.mark.parametrize('_format',
        (
        'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_col_idx1', (0, 1, 2))
    def test_accuracy(
        self, _X_factory, _dtype, _format, _col_idx1, _shape, _columns
    ):

        # coo, dia, & bsr matrix/array are blocked. should raise here.

        if _dtype == 'str' and _format not in ('np', 'pd'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _X_wip = _X_factory(
            _format=_format,
            _dtype=_dtype,
            _has_nan=False,
            _shape=_shape
        )

        if _format == 'np':
            _X_base = _X_wip.copy()
        elif _format == 'pd':
            _X_base = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _X_base = _X_wip.toarray()
        else:
            raise Exception


        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
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
            if _format not in ('np', 'pd'):
                new_column1 = np.zeros(_shape[0]).astype(np.float64)
                new_column1[column1[:len(column1)//2].astype(np.int32)] = \
                    column1[len(column1)//2:]
                column1 = new_column1
                del new_column1


            if _dtype == 'flt':
                assert np.array_equal(column1, _X_base[:, _col_idx1], equal_nan=True)
            elif _dtype == 'str':
                assert np.array_equal(
                    column1.astype(str),
                    _X_base[:, _col_idx1].astype(str)
                )




