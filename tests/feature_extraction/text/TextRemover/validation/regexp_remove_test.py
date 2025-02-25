# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextRemover._validation._regexp_remove \
    import _val_regexp_remove




class TestValRegExpRemove:


    @pytest.mark.parametrize('junk_single_regexp',
        (-2.7, -1, 0, 1, 2.7, True, False, {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_regexp_as_single(self, junk_single_regexp):
        # could be None, str, or re.Pattern
        with pytest.raises(TypeError):
            _val_regexp_remove(junk_single_regexp, list('abcde'))


    @pytest.mark.parametrize('good_single_regexp',
        (None, '[a-m]*', re.compile('[a-m]+'))
    )
    def test_accepts_good_single(self, good_single_regexp):
        # could be None, str, re.Pattern

        _val_regexp_remove(good_single_regexp, list('abcde'))


    @pytest.mark.parametrize('junk_seq_regexp', ([True, '[a-q]'], ('[1-3]', 0)))
    def test_rejects_junk_regexp_as_seq(self, junk_seq_regexp):
        # could be str, re.Pattern, or list[of the 2]

        with pytest.raises(TypeError):
            _val_regexp_remove(junk_seq_regexp, list('ab'))


    def test_rejects_bad_regexp_as_seq(self):

        # too long
        with pytest.raises(ValueError):
            _val_regexp_remove(['\W' for _ in range(6)], list('abcde'))


        # too short
        with pytest.raises(ValueError):
            _val_regexp_remove(['\W' for _ in range(4)], list('abcde'))


    def test_accepts_good_sequence(self):

        for trial in range(20):

            _regexp = np.random.choice(
                [re.compile('a{0, 2}'), '\W', '\d', re.compile('[a-d]*'),
                 re.compile('[a-q]*'), False],
                (5, ),
                replace=True
            ).tolist()

            _val_regexp_remove(
                _regexp,
                _X = list('abcde')
            )






