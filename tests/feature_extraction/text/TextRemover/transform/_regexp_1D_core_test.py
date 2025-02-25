# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextRemover._transform._regexp_1D_core \
    import _regexp_1D_core



class TestRegExp1DCore:

    # no validation



    def test_accuracy(self):


        # False skips -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = \
            _regexp_1D_core(X, [False, False, False, 'd', 'e'], None)

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abc'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, False, False])
        # END False skips -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abCde')
        out_X, out_mask = _regexp_1D_core(X, 'c', re.I)

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, False, True, True])
        # END str, one match -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('cbadC')
        out_X, out_mask = _regexp_1D_core(X, 'c', re.I)

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bad'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, True, True, False])

        X = list('cbadc')
        out_X, out_mask = _regexp_1D_core(X, re.compile('c'), None)

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bad'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, True, True, False])
        # END str, two matches -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        X = list('abcde')
        out_X, out_mask = _regexp_1D_core(
            X,
            ['A|E', False, '\w', 'abc', False],
            re.I
        )

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, False, True, True])

        X = list('abcde')
        out_X, out_mask = _regexp_1D_core(
            X,
            ['a|e', False, '\w', re.compile('abc'), False],
            None
        )

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('bde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, False, True, True])

        X = list('abcde')
        out_X, out_mask = _regexp_1D_core(
            X,
            [re.compile('A|E'), False, 'C', 'ABC', False],
            [None, False, None, None, False]
        )

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abcde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        X = list('abcde')
        out_X, out_mask = _regexp_1D_core(
            X,
            list('EDCBA'),
            [None, None, re.I, None, None]
        )

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, list('abde'))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, False, True, True])

        X = list('zzzzz')
        out_X, out_mask = _regexp_1D_core(
            X,
            re.compile('z'),
            None
        )

        assert isinstance(out_X, list)
        assert np.array_equal(out_X, [])

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, False, False, False, False])
        # END lists -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --












