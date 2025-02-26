# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextReplacer._validation._str_replace \
    import _val_str_replace



class TestValStrReplace:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return 10


    @staticmethod
    @pytest.fixture(scope='function')
    def _text(_shape):
        return np.random.choice(list('abcde'), (_shape, ), replace=True)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('junk_sr',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_str_replace(self, _text, junk_sr):
        # could be None, tuple, set, list
        with pytest.raises(TypeError):
            _val_str_replace(junk_sr, _text)


    def test_rejects_bad_len_str_replace(self, _text):

        # too short
        with pytest.raises(ValueError):
            _val_str_replace([('@', '') for _ in range(len(_text)-1)], _text)

        # too long
        with pytest.raises(ValueError):
            _val_str_replace([('@', '') for _ in range(len(_text)+1)], _text)


    @pytest.mark.parametrize('good_sr',
        (
            None, ('a', ''), ('a', '', 1), {('b', 'B'), ('c', 'C')},
            [('@', '') for _ in range(10)], [('@', '', 1) for _ in range(10)]
        )
    )
    def test_accepts_good_str_replace(self, _text, good_sr):
        # could be None, tuple, set, list
        _val_str_replace(good_sr, _text)






