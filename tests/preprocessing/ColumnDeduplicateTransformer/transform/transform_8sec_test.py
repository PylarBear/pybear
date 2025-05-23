# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd

from pybear.preprocessing._ColumnDeduplicateTransformer._transform._transform \
    import _transform

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer



@pytest.mark.parametrize('_dtype', ('flt', 'str'), scope='module')
@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
@pytest.mark.parametrize('_keep', ('first', 'last', 'random'), scope='module')
@pytest.mark.parametrize('_equal_nan', (True, False), scope='module')
class TestTransform:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl_equal_nan_is_True():
        return [
            [0, 3],
            [2, 6, 8]
        ]

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix',
            'dia_matrix', 'lil_matrix', 'dok_matrix', 'csr_array',
            'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array'
        )
    )
    def test_output(
        self, _X_factory, _dtype, _format, _shape, _master_columns, _equal_nan,
        _has_nan, _keep, _dupl_equal_nan_is_True
    ):

        # rejects everything except np.ndarray, pd.DataFrame,
        # and scipy csc_matrix/csc_array. should except.

        if _dtype == 'str' and _format not in ['np', 'pd']:
            pytest.skip(reason=f"scipy sparse cant take strings")

        # BUILD X v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        _X_wip = _X_factory(
            _dupl=_dupl_equal_nan_is_True,
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _shape=_shape
        )
        # END BUILD_X v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # assign the correct column mask - - - - - - - - - - - -
        # when not equal_nan, all columns must be unequal (full mask, all
        # columns are kept) because every column in X gets some nans
        if _equal_nan:
            _del = []
            for _set in _dupl_equal_nan_is_True:
                if _keep == 'first':
                    _del += _set[1:]
                elif _keep == 'last':
                    _del += _set[:-1]
                elif _keep == 'random':
                    __ = _set.copy()
                    __.remove(np.random.choice(__))
                    _del += __

            _del.sort()

            _column_mask = \
                np.array([i not in _del for i in range(_shape[1])]).astype(bool)
        elif not _equal_nan:
            _column_mask = np.array([True for i in range(_shape[1])]).astype(bool)
            assert np.sum(_column_mask) == _shape[1]
        else:
            raise Exception
        # END assign the correct column mask - - - - - - - - - - - -

        # retain the original dtype(s)
        _og_format = type(_X_wip)
        if isinstance(_X_wip, pd.core.frame.DataFrame):
            _og_dtype = _X_wip.dtypes
        else:
            _og_dtype = _X_wip.dtype

        # retain the original num columns
        _og_cols = _X_wip.shape[1]

        # apply the correct column mask to the original X
        if _format in ('np', 'pd', 'csc_matrix', 'csc_array'):
            out = _transform(_X_wip, _column_mask)
        else:
            with pytest.raises(AssertionError):
                _transform(_X_wip, _column_mask)
            pytest.skip(reason=f"cant do more tests after exception")


        # ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # output format is same as given
        assert isinstance(out, _og_format)

        # output dtypes are same as given
        if isinstance(out, pd.core.frame.DataFrame):
            assert np.array_equal(out.dtypes, _og_dtype[_column_mask])
        else:
            assert out.dtype == _og_dtype

        # out matches _column_mask
        assert out.shape[1] == sum(_column_mask)

        _kept_idxs = np.arange(_shape[1])[_column_mask]

        for _new_idx, _kept_idx in enumerate(_kept_idxs, 0):

            # nans in string columns are being a real pain
            # _parallel_column_comparer instead of np.array_equal

            if isinstance(_X_wip, np.ndarray):
                _out_col = out[:, _new_idx]
                _og_col = _X_wip[:, _kept_idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _out_col = out.iloc[:, _new_idx]
                _og_col = _X_wip.iloc[:, _kept_idx]
            else:
                _out_col = out.tocsc()[:, [_new_idx]].toarray()
                _og_col = _X_wip.tocsc()[:, [_kept_idx]].toarray()


            if not _has_nan or (_has_nan and _equal_nan):
                assert _parallel_column_comparer(
                    _out_col, _og_col, 1e-5, 1e-8, _equal_nan
                )
            else:
                assert not _parallel_column_comparer(
                    _out_col, _og_col, 1e-5, 1e-8, _equal_nan
                )

        # END ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **








