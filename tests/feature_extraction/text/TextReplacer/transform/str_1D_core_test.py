# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._str_1D_core \
    import _str_1D_core



class TestStr1DCore:



    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], (1,), {1,2}, {'A':1},
         set(list('abcde')), tuple(list('abcde')), lambda x: x)
    )
    def test_blocks_junk_X(self, junk_X):

        with pytest.raises(AssertionError):
            _str_1D_core(junk_X, [set(('a', ''),), set(('b', 'B'),)])


    # pizza there is currently no validation on sr
    # @pytest.mark.parametrize('junk_sr',
    #     (-2.7, -1, 0, 1, 2.7, True, None, [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    # )
    # def test_blocks_junk_sr(self, junk_sr):
    #
    #     with pytest.raises(AssertionError):
    #         _str_1D_core(list('ab'), junk_sr)


    def test_takes_good_X_and_sr(self):
        _str_1D_core(list('ab'), [set((('a', ''), )), set((('b', 'B'),))])

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    def test_accuracy(self):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all empty makes no changes
        X = list('abcde')
        _str_replace = [set() for _ in range(len(X))]

        out = _str_1D_core(X, _str_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, X)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # single replace
        X = list('abcde')
        _str_replace = [set((tuple(('a', '')),)) for _ in range(len(X))]

        out = _str_1D_core(X, _str_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, [''] + list('bcde'))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # two replaced
        X = list('abcde')
        _str_replace = [
            set(),
            set(),
            set(),
            set((tuple(('z', '')), tuple(('d', '')))),
            set((tuple(('e', '')),))
        ]

        out = _str_1D_core(X, _str_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, ['a', 'b', 'c', '', ''])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # no matches
        X = list('abcde')
        _str_replace = [
            set((tuple(('q', '')),)),
            set((tuple(('r', '')),)),
            set((tuple(('s', '')),)),
            set((tuple(('t', '')),)),
            set((tuple(('u', '')),))
        ]

        out = _str_1D_core(X, _str_replace)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcde'))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








