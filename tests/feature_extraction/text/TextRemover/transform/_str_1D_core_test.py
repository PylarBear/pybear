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
        out = _str_1D_core(X, [False, False, False, 'd', 'e'])
        assert isinstance(out, list)
        assert np.array_equal(out, list('abc'))
        # END False skips -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out = _str_1D_core(X, 'c')
        assert isinstance(out, list)
        assert np.array_equal(out, list('abde'))
        # END str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('cbadc')
        out = _str_1D_core(X, 'c')
        assert isinstance(out, list)
        assert np.array_equal(out, list('bad'))
        # END str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # set, one match -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('cbadc')
        out = _str_1D_core(X, {'a', 'e', 'f'})
        assert isinstance(out, list)
        assert np.array_equal(out, list('cbdc'))
        # END set, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # set, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out = _str_1D_core(X, {'a', 'e'})
        assert isinstance(out, list)
        assert np.array_equal(out, list('bcd'))
        # END set, two matches -- -- -- -- -- -- -- -- -- -- -- -- --

        # lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out = _str_1D_core(X, [{'a', 'e'}, False, 'c', 'abc', False])
        assert isinstance(out, list)
        assert np.array_equal(out, list('bde'))

        X = list('abcde')
        out = _str_1D_core(X, [{'A', 'E'}, False, 'C', 'ABC', False])
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcde'))

        X = list('abcde')
        out = _str_1D_core(X, list('edcba'))
        assert isinstance(out, list)
        assert np.array_equal(out, list('abde'))

        X = list('zzzzz')
        out = _str_1D_core(X, 'z')
        assert isinstance(out, list)
        assert np.array_equal(out, [])
        # END lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

