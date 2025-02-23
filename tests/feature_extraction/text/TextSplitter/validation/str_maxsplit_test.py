# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter._validation._str_maxsplit \
    import _val_str_maxsplit



class TestValStrMaxSplit:

    # can be single integer, None, or list of integers, Nones, and Falses

    @pytest.mark.parametrize('junk_single_sms',
        (-2.7, 2.7, True, False, 'trash', {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_single_sms(self, junk_single_sms):

        with pytest.raises(TypeError):
            _val_str_maxsplit(junk_single_sms, list('abcde'))


    @pytest.mark.parametrize('junk_ms',
         ([-2.7, 2.7], list('ab'), (True, False), tuple(list('12')))
    )
    def test_rejects_junk_sms_sequence(self, junk_ms):

        with pytest.raises(TypeError):
            _val_str_maxsplit(junk_ms, list('ab'))


    @pytest.mark.parametrize('bad_ms_len', (2,3,4,6,7,8))
    def test_rejects_bad_sms_len(self, bad_ms_len):

        with pytest.raises(ValueError):
            _val_str_maxsplit(list(range(1,9))[:bad_ms_len], list('abcde'))


    def test_accepts_single_None_single_int(self):

        _val_str_maxsplit(None, list('abcde'))

        _val_str_maxsplit(-10_000, list('abcde'))

        _val_str_maxsplit(0, list('abcde'))

        _val_str_maxsplit(10_000, list('abcde'))


    def test_accepts_good_sequence(self):

        # can contain integers, Nones, and Falses

        for trial in range(20):

            _maxsplit = np.random.choice(
                [int(np.random.randint(-10000, 10000)), None, False],
                (5,),
                replace=True
            ).tolist()

            _val_str_maxsplit(
                _str_maxsplit=_maxsplit,
                _X = list('abcde')
            )




