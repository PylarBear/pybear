# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# IMPORTANT!
# this is sparse so this uses _parallel_ss_comparer!

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit import (
    _parallel_ss_comparer as pssc,
    _column_getter as cg
)

import numpy as np
import scipy.sparse as ss

import pytest




class TestSSColumnComparer:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (1000, 2)


    # np cant be int if using nans
    # _column_getter cant take coo, dia, bsr
    @pytest.mark.parametrize('_format', ('csc', 'csr', 'lil', 'dok'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_accuracy(
        self, _X_factory, _format, _has_nan, _equal_nan, _shape
    ):

        # a sneaky trick here. _X_factory peppers nans after propagating
        # duplicates. which means nans are likely to be different on every
        # column. so if create a 2 column array and both columns are the
        # same, then both will be identical except for the nans.


        _X_flt = _X_factory(
            _dupl=[[0,1]],
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.33,
            _shape=_shape
        )

        if _format == 'csc':
            _X_flt = ss.csc_array(_X_flt)
        elif _format == 'csr':
            _X_flt = ss.csr_array(_X_flt)
        elif _format == 'lil':
            _X_flt = ss.lil_array(_X_flt)
        elif _format == 'dok':
            _X_flt = ss.dok_matrix(_X_flt)
        else:
            raise Exception

        _X1 = cg._column_getter(_X_flt, 0)
        _X2 = cg._column_getter(_X_flt, 1)

        assert isinstance(_X1, np.ndarray)
        assert isinstance(_X2, np.ndarray)

        _are_equal = pssc._parallel_ss_comparer(
            _X1, _X2, _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan
        )

        if _equal_nan and not _has_nan:
            assert _are_equal
        elif _equal_nan and _has_nan:
            assert _are_equal
        elif not _equal_nan and not _has_nan:
            assert _are_equal
        elif not _equal_nan and _has_nan:
            assert not _are_equal
        else:
            raise Exception
























