# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# IMPORTANT!
# this is sparse so this uses _parallel_ss_comparer!

from pybear.preprocessing.SlimPolyFeatures._partial_fit import (
    _parallel_ss_comparer as pssc,
    _columns_getter as cg
)

import numpy as np
import scipy.sparse as ss

import pytest



pytest.skip(reason=f"pizza says this goes away with explody ss in _columns_getter()", allow_module_level=True)




class TestSSColumnComparer:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (1000, 2)


    # np cant be int if using nans
    @pytest.mark.parametrize('_format', ('csc', 'csr', 'coo', 'bsr'))
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
        elif _format == 'coo':
            _X_flt = ss.coo_array(_X_flt)
        elif _format == 'bsr':
            _X_flt = ss.bsr_matrix(_X_flt)
        else:
            raise Exception

        _X1 = cg._columns_getter(_X_flt, 0)
        _X2 = cg._columns_getter(_X_flt, 1)

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
























