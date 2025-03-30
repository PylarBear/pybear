# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._regexp_1D_core \
    import _regexp_1D_core



class TestRegExp1DCore:



    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], (1,), {1,2}, {'A':1},
         set(list('abcde')), tuple(list('abcde')), lambda x: x)
    )
    def test_blocks_junk_X(self, junk_X):

        with pytest.raises(AssertionError):
            _regexp_1D_core(junk_X, [set(('a', ''),), set(('b', 'B'),)])


    # pizza there is currently no validation for this on rr
    # @pytest.mark.parametrize('junk_rr',
    #     (-2.7, -1, 0, 1, 2.7, True, None, [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    # )
    # def test_blocks_junk_rr(self, junk_rr):
    #
    #     with pytest.raises(AssertionError):
    #         _regexp_1D_core(list('ab'), junk_rr)
    #
    #
    def test_takes_good_X_and_rr(self):
        _regexp_1D_core(list('ab'), [set((('[a-m]', ''), )), set((('[b-d]', 'B'),))])

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    def test_accuracy(self):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all empty makes no changes
        X = list('abcde')
        _regexp_replace = [set() for _ in range(len(X))]

        out = _regexp_1D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, X)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # single replace
        X = list('abcde')
        _regexp_replace = [set((tuple((re.compile('a'), '')),)) for _ in range(len(X))]

        out = _regexp_1D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, [''] + list('bcde'))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # two replaced
        X = list('abcDE')
        _regexp_replace = [
            set(),
            set(),
            set(),
            set((tuple((re.compile('d', re.I), '')),)),
            set((tuple(('q', '', 1, re.I)), tuple(('e', '', 1, re.I))))
        ]

        out = _regexp_1D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, ['a', 'b', 'c', '', ''])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # no matches
        X = list('abcde')
        _regexp_replace = [
            set((tuple((re.compile('q', re.I), '')),)),
            set((tuple(('r', '', 2, re.X)),)),
            set((tuple(('s', '')),)),
            set((tuple(('t', '', 1)),)),
            set((tuple((re.compile('u'), '')),))
        ]

        out = _regexp_1D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcde'))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








