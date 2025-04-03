# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter._validation._maxsplit \
    import _val_maxsplit



class TestValMaxSplit:

    # can be single integer, None, or list of integers and/or Nones

    @pytest.mark.parametrize('junk_single_ms',
        (-2.7, 2.7, True, False, 'trash', {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_single_ms(self, junk_single_ms):

        with pytest.raises(TypeError):
            _val_maxsplit(junk_single_ms, 5)


    @pytest.mark.parametrize('junk_seq_ms',
         ([-2.7, 2.7], list('ab'), (True, False), tuple(list('12')))
    )
    def test_rejects_junk_ms_sequence(self, junk_seq_ms):

        with pytest.raises(TypeError):
            _val_maxsplit(junk_seq_ms, 2)


    @pytest.mark.parametrize('bad_ms_len', (2,3,4,6,7,8))
    def test_rejects_bad_ms_len(self, bad_ms_len):

        with pytest.raises(ValueError):
            _val_maxsplit(list(range(1,9))[:bad_ms_len], 5)


    def test_accepts_single_None_single_int(self):

        _val_maxsplit(None, 5)

        _val_maxsplit(-10_000, 5)

        _val_maxsplit(0, 5)

        _val_maxsplit(10_000, 5)


    def test_accepts_good_sequence(self):

        # can contain integers and/or Nones

        for trial in range(20):

            _maxsplit = np.random.choice(
                [int(np.random.randint(-10000, 10000)), None, None],
                (5,),
                replace=True
            ).tolist()

            _val_maxsplit(
                _maxsplit=_maxsplit,
                _n_rows = 5
            )




