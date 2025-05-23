# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates



class TestNpFindDuplicates:


    @pytest.mark.parametrize('_format',
        (
         'coo_matrix', 'dia_matrix', 'bsr_matrix',
         'coo_array', 'dia_array', 'bsr_array'
        )
    )
    def test_blocks_ss_coo_dia_bsr(self, _X_factory, _format, _shape, _dupl1):

        _X_wip = _X_factory(
            _dupl=_dupl1,
            _format=_format,
            _dtype='flt',
            _has_nan=False,
            _columns=None,
            _zeros=0.25,
            _shape=_shape
        )

        with pytest.raises(AssertionError):
            _find_duplicates(
                _X_wip,
                _rtol=1e-5,
                _atol=1e-8,
                _equal_nan=True,
                _n_jobs=1   # leave set a 1 because of confliction
            )


    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_dupl_set', (1, 2, 3, 4))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_format',
    (
     'np', 'csc_matrix', 'csc_array',
     # 'csr_matrix', 'lil_matrix',
     # 'dok_matrix', 'csr_array', 'lil_array', 'dok_array'    # pizza
    )
    )
    def test_accuracy(
        self, _X_factory, _dtype, _equal_nan, _format, _shape, _has_nan,
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

        _X_wip = _X_factory(
            _dupl=_dupl,
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.25,
            _shape=_shape
        )

        # leave n_jobs set a 1 because of confliction
        out = _find_duplicates(
            _X_wip, _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan, _n_jobs=1
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


    # pizza
    @pytest.mark.parametrize('_format',
        (
         'csc_matrix', 'csc_array',
         #'csr_matrix', 'lil_matrix', 'dok_matrix',
         #'csr_array', 'lil_array', 'dok_array'
        )
    )
    def test_accuracy_ss_all_zeros(self, _X_factory, _format, _shape):

        _X_wip = _X_factory(
            _dupl=[list(range(_shape[1]))],
            _format=_format,
            _dtype='flt',
            _has_nan=False,
            _constants={0: 0},
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        assert np.sum(_X_wip.toarray()) == 0

        # leave set a 1 because of confliction
        out = _find_duplicates(
            _X_wip, _rtol=1e-5, _atol=1e-8, _equal_nan=True, _n_jobs=1
        )

        assert np.array_equal(out[0], list(range(_shape[1])))





