# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._partial_fit. _column_getter \
    import _column_getter

import uuid
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
    def test_accuracy(self, _has_nan, _dtype, _format, _col_idx1, _shape):

        # coo, dia, & bsr matrix/array are blocked. should raise here.

        if _dtype == 'str' and _format not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        if _dtype == 'num':
            _X = np.random.uniform(0, 1, _shape)
        elif _dtype == 'str':
            _X = np.random.choice(list('abcdefghijkl'), _shape, replace=True)
        else:
            raise Exception

        if _format == 'df':
            _X = pd.DataFrame(
                data=_X,
                columns=[str(uuid.uuid4())[:5] for _ in range(_shape[1])]
            )

        if _has_nan:
            if _format == 'df':
                _nan_pool = [np.nan, pd.NA, None, 'nan', 'NaN', 'NAN', '<NA>']
                for _c_idx in range(_shape[1]):
                    _idxs = np.random.choice(range(_shape[0]), _shape[0] // 5)
                    _values = np.random.choice(_nan_pool, _shape[0] // 5)
                    _X.iloc[_idxs, _c_idx] = _values
                    del _values
                del _nan_pool
            elif _format == 'ndarray':  # np and ss only take np.nan
                for _c_idx in range(_shape[1]):
                    _idxs = np.random.choice(range(_shape[0]), _shape[0] // 5)
                    _X[_idxs, _c_idx] = np.nan
            else:
                for _c_idx in range(_shape[1]):
                    _idxs = np.random.choice(range(_shape[0]), _shape[0] // 5)
                    _X[_idxs, [_c_idx]] = np.nan

            del _idxs

        if _format == 'ndarray':
            _X_wip = _X
        elif _format == 'df':
            _X_wip = _X
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

        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array,
             ss.dia_matrix, ss.dia_array,
             ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _column_getter(_X_wip, _col_idx1)
            pytest.skip(reason=f"cant do more tests after exception")
        else:
            column1 = _column_getter(_X_wip, _col_idx1)

        assert len(column1.shape) == 1

        # if running scipy sparse, then column1 will be the 'data' attr.
        # take it easy on yourself, just use the ss csc, get the data
        # attr and compare against _column_getter to ensure the correct
        # column is being pulled
        if _format == 'ndarray':
            og_col = _X_wip[:, _col_idx1]
        elif _format == 'df':
            og_col = _X_wip.iloc[:, _col_idx1]
        else:
            og_col = _X_wip.tocsc()[:, [_col_idx1]].toarray().ravel()


        if _dtype == 'num':
            assert np.array_equal(
                column1.astype(np.float64),
                og_col.astype(np.float64),
                equal_nan=True
            )
        elif _dtype == 'str':
            assert np.array_equal(
                column1.astype(str),
                og_col.astype(str)
            )























