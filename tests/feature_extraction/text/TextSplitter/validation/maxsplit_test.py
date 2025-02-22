# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextSplitter._validation._maxsplit import \
    _val_maxsplit




class TestValMaxSplit:


    @pytest.mark.parametrize('junk_ms',
        (-2.7, 2.7, True, False, 'trash', {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_ms(self, junk_ms):

        with pytest.raises(TypeError):
            _val_maxsplit(junk_ms, list('abcde'))


    @pytest.mark.parametrize('junk_ms', ([-2.7, 2.7], list('ab'), (True, False)))
    def test_rejects_junk_ms_sequence(self, junk_ms):

        with pytest.raises(TypeError):
            _val_maxsplit(junk_ms, list('ab'))


    @pytest.mark.parametrize('bad_ms_len', (2,3,4,6,7,8))
    def test_rejects_bad_ms_len(self, bad_ms_len):

        with pytest.raises(ValueError):
            _val_maxsplit(list(range(1,9))[:bad_ms_len], list('abcde'))


    def test_accepts_None(self):

        _val_maxsplit(
            None,
            list('abcde')
        )


    @pytest.mark.parametrize('value', (-2, -1, 0, 1, 2))
    def test_accepts_good(self, value):

        _X = list('abcde')

        _val_maxsplit(
            value,
            _X
        )

        _val_maxsplit(
            [value for _ in range(len(_X))],
            _X
        )




