# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._transform._manage_wrap_idxs \
    import _manage_wrap_idxs

import pytest

import numbers



class TestManageWrapIdxs:


    @staticmethod
    @pytest.fixture(scope='module')
    def _first_line():
        return ['CHALLENGER', 'DEEP', 'FRIED']


    @staticmethod
    @pytest.fixture(scope='module')
    def _second_line():
        return ['EGG', 'SALAD', 'SHOOTER']



    def test_accuracy(self, _first_line, _second_line):

        # def _manage_wrap_idxs(
        #     _first_line: Sequence[str],
        #     _first_hits: Sequence[int],
        #     _second_hits: Sequence[int],
        #     _n_len: numbers.Integral
        # ) -> tuple[int, int]:


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['CHALLENGER', 'DEEP']
        _first_hits = [0]
        _second_hits = []
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 2
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 1

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['DEEP', 'FRIED']
        _first_hits = [1]
        _second_hits = []
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 3
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 1

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['FRIED', 'EGG']
        _first_hits = []
        _second_hits = []
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 2
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 1

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['EGG', 'SALAD']
        _first_hits = []
        _second_hits = [0]
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 2
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 0

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['SALAD', 'SHOOTER']
        _first_hits = []
        _second_hits = [1]
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 2
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 1

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['CHALLENGER', 'DEEP', 'FRIED']
        _first_hits = [0]
        _second_hits = []
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 3
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 2

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _ngram = ['EGG', 'SALAD', 'SHOOTER']
        _first_hits = []
        _second_hits = [0]
        _n_len = len(_ngram)
        first_idx, second_idx = _manage_wrap_idxs(
            _first_line, _first_hits, _second_hits, _n_len
        )

        assert isinstance(first_idx, numbers.Integral)
        assert first_idx == 1
        assert isinstance(second_idx, numbers.Integral)
        assert second_idx == 0

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --