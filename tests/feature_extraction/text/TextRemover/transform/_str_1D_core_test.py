# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextRemover._transform._str_1D_core import \
    _str_1D_core



class TestStr1DCore:

    # no validation



    def test_accuracy(self):


        # False skips -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, [False, False, False, 'd', 'e'])

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abc'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, False, False])
        # END False skips -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, 'c')

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, False, True, True])
        # END str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('cbadc')
        out_X, out_mask = _str_1D_core(X, 'c')

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bad'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, True, True, False])
        # END str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # set, one match -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('cbadc')
        out_X, out_mask = _str_1D_core(X, {'a', 'e', 'f'})

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('cbdc'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, False, True, True])
        # END set, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # set, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, {'a', 'e'})

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bcd'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, True, True, False])
        # END set, two matches -- -- -- -- -- -- -- -- -- -- -- -- --

        # lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, [{'a', 'e'}, False, 'c', 'abc', False])

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, False, True, True])

        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, [{'A', 'E'}, False, 'C', 'ABC', False])

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abcde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        X = list('abcde')
        out_X, out_mask = _str_1D_core(X, list('edcba'))

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, False, True, True])

        X = list('zzzzz')
        out_X, out_mask = _str_1D_core(X, 'z')

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, [])

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, False, False, False, False])
        # END lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --





