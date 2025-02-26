# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy
import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._regexp_2D_core \
    import _regexp_2D_core



class TestRegExp2DCore:



    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], (1,), {1,2}, {'A':1},
         list('abcde'), set(list('abcde')), tuple(list('abcde')), lambda x: x)
    )
    def test_blocks_junk_X(self, junk_X):

        with pytest.raises(AssertionError):
            _regexp_2D_core(junk_X, [set(('a', ''),), set(('b', 'B'),)])


    @pytest.mark.parametrize('junk_rr',
        (-2.7, -1, 0, 1, 2.7, True, None, [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    )
    def test_blocks_junk_rr(self, junk_rr):

        X = [list('ab'), list('xyz')]

        with pytest.raises(AssertionError):
            _regexp_2D_core(X, junk_rr)


    def test_takes_good_X_and_rr(self):

        X = [list('ab'), list('xyz')]
        rr = [set((('[a-m]', ''), )), set((('[b-d]', 'B'),))]
        _regexp_2D_core(X, rr)

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    def test_accuracy(self):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all empty makes no changes
        X = [list('abcde'), list('wxyz')]
        _regexp_replace = [set() for _ in range(len(X))]

        out = _regexp_2D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, X))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # single replace
        X = [list('abcde'), list('123')]
        _regexp_replace = [set((tuple((re.compile('a'), '')),)) for _ in range(len(X))]

        out = _regexp_2D_core(X, _regexp_replace)
        exp = deepcopy(X)
        exp[0][0] = ''

        assert isinstance(out, list)
        assert all(map(np.array_equal, out, exp))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # two replaced
        X = [list('abcDE'), list('defgh')]
        _regexp_replace = [
            set((tuple((re.compile('d', re.I), '')),)),
            set((tuple(('e', '', 1, re.I)),))
        ]

        out = _regexp_2D_core(X, _regexp_replace)
        exp = deepcopy(X)
        exp[0][3] = ''
        exp[1][1] = ''

        assert isinstance(out, list)
        assert all(map(np.array_equal, out, exp))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # no matches
        X = [list('abcde') for _ in range(5)]
        _regexp_replace = [
            set((tuple((re.compile('q', re.I), '')),)),
            set((tuple(('r', '', 2, re.X)),)),
            set((tuple(('s', '')),)),
            set((tuple(('t', '', 1)),)),
            set((tuple((re.compile('u'), '')),))
        ]

        out = _regexp_2D_core(X, _regexp_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, X))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








