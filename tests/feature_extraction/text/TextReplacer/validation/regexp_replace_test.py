# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._validation._regexp_replace \
    import _val_regexp_replace



class TestValRegExpReplace:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return 10


    @staticmethod
    @pytest.fixture(scope='function')
    def _text(_shape):
        return np.random.choice(list('abcde'), (_shape, ), replace=True)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('junk_rr',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_regexp_replace(self, _text, junk_rr):
        # could be None, tuple, set, list
        with pytest.raises(TypeError):
            _val_regexp_replace(junk_rr, _text)


    def test_rejects_bad_len_regexp_replace(self, _text):

        # too short
        with pytest.raises(ValueError):
            _val_regexp_replace([('@', '') for _ in range(len(_text)-1)], _text)

        # too long
        with pytest.raises(ValueError):
            _val_regexp_replace([('@', '') for _ in range(len(_text)+1)], _text)


    @pytest.mark.parametrize('good_rr',
        (
            None, ('a', ''), (re.compile('a'), '', 1),
            {('b', 'B'), (re.compile('c', re.X), 'C')},
            [(re.compile('@'), '', 2) for _ in range(10)],
            [('@', '', 1, re.I) for _ in range(10)],
            [False for _ in range(10)]
        )
    )
    def test_accepts_good_regexp_replace(self, _text, good_rr):
        # could be None, tuple, set, list
        _val_regexp_replace(good_rr, _text)










