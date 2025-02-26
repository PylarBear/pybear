# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._str_2D_core \
    import _str_2D_core



class TestStr2DCoreTest:

    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0, 1], (1,), {1, 2}, {'A': 1},
        set(list('abcde')), tuple(list('abcde')), lambda x: x)
    )
    def test_blocks_junk_X(self, junk_X):
        with pytest.raises(AssertionError):
            _str_2D_core(junk_X, [set(('a', ''), ), set(('b', 'B'), )])


    @pytest.mark.parametrize('junk_sr',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {1, 2}, {'A': 1}, lambda x: x)
    )
    def test_blocks_junk_sr(self, junk_sr):
        with pytest.raises(AssertionError):
            _str_2D_core([list('ab'), list('xyz')], junk_sr)


    def test_takes_good_X_and_sr(self):
        _str_2D_core(
            [list('ab'), list('xyz')],
            [set((('a', ''),)), set((('b', 'B'),))]
        )

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def test_accuracy(self):
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all empty makes no changes
        X = [list('xyz'), list('abcde')]
        _str_replace = [set() for _ in range(len(X))]

        out = _str_2D_core(X, _str_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, X))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # single replace
        X = [list('qrs'), list('abcde')]
        _str_replace = [set((tuple(('a', '')),)) for _ in range(len(X))]

        out = _str_2D_core(X, _str_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, [list('qrs'), [''] + list('bcde')]))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # two replaced
        X = [list('abcde'), list('xyz')]
        _str_replace = [
            set((tuple(('d', '')),)),
            set((tuple(('d', '')), tuple(('x', ''))))
        ]

        out = _str_2D_core(X, _str_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, [['a','b','c','','e'], ['', 'y', 'z']]))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # no matches
        X = [list('abc'), list('123')]
        _str_replace = [
            set((tuple(('q', '')),)),
            set((tuple(('r', '')),))
        ]

        out = _str_2D_core(X, _str_replace)
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, [list('abc'), list('123')]))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --






