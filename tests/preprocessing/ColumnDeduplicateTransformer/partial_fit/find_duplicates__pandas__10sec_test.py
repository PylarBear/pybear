# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates



class TestPdFindDuplicates:


    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_dupl_set', (1,2,3,4))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_accuracy(
        self, _X_factory, _dtype, _equal_nan, _shape, _has_nan, _master_columns,
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
            _format='pd',  # pizza pl?
            _dtype=_dtype,
            _has_nan=_has_nan,
            _columns=_master_columns.copy()[:_shape[1]],
            _zeros=0.25,
            _shape=_shape
        )

        # leave n_jobs set at 1 because of confliction
        out = _find_duplicates(
            _X, _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan, _n_jobs=1
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





