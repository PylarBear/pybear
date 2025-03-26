# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

import numpy as np

from pybear.feature_extraction.text._NGramMerger._transform._get_wrap_match_idxs \
    import _get_wrap_match_idxs


import pytest




class TestGetWrapMatchIdxs:

    # def _get_wrap_match_idxs(
    #     _first_line: Sequence[str],
    #     _start_idx: numbers.Integral,
    #     _end_idx: numbers.Integral,
    #     _wrap_match_idx: Sequence[numbers.Integral],
    #     _n_len: numbers.Integral
    # ) -> tuple[list[int], list[int]]:


    @staticmethod
    @pytest.fixture(scope='module')
    def _first_line():
        return ['CHALLENGER', 'DEEP', 'FRIED']


    @staticmethod
    @pytest.fixture(scope='module')
    def _second_line():
        return ['EGG', 'SALAD', 'SHOOTER']

    
    def test_accuracy(self, _first_line, _second_line):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['DEEP', 'FRIED', 'EGG']
        _start_idx = 1
        _end_idx = 2
        _wrap_match_idx = [0]
        _n_len = len(_ngram)

        first_line_idxs, second_line_idxs = _get_wrap_match_idxs(
            _first_line,
            _start_idx,
            _end_idx,
            _wrap_match_idx,
            _n_len
        )

        assert isinstance(first_line_idxs, Sequence)
        assert all(map(
            isinstance,
            first_line_idxs,
            (numbers.Integral for _ in first_line_idxs)
        ))
        assert all(map(lambda x: x >=0 , first_line_idxs))
        assert np.array_equal(first_line_idxs, [1, 2])

        assert isinstance(second_line_idxs, Sequence)
        assert all(map(
            isinstance,
            second_line_idxs,
            (numbers.Integral for _ in second_line_idxs)
        ))
        assert all(map(lambda x: x >=0 , second_line_idxs))
        assert np.array_equal(second_line_idxs, [0])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['FRIED', 'EGG', 'SALAD']
        _start_idx = 1
        _end_idx = 2
        _wrap_match_idx = [1]
        _n_len = len(_ngram)

        first_line_idxs, second_line_idxs = _get_wrap_match_idxs(
            _first_line,
            _start_idx,
            _end_idx,
            _wrap_match_idx,
            _n_len
        )

        assert isinstance(first_line_idxs, Sequence)
        assert all(map(
            isinstance,
            first_line_idxs,
            (numbers.Integral for _ in first_line_idxs)
        ))
        assert all(map(lambda x: x >=0 , first_line_idxs))
        assert np.array_equal(first_line_idxs, [2])

        assert isinstance(second_line_idxs, Sequence)
        assert all(map(
            isinstance,
            second_line_idxs,
            (numbers.Integral for _ in second_line_idxs)
        ))
        assert all(map(lambda x: x >=0 , second_line_idxs))
        assert np.array_equal(second_line_idxs, [0, 1])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --





