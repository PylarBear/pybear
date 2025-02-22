# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextSplitter._validation._regexp import \
    _val_regexp




class TestValSep:


    @pytest.mark.parametrize('junk_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_regexp_as_single(self, junk_sep):
        # could be None, str, or re.Pattern
        with pytest.raises(TypeError):
            _val_regexp(junk_sep, list('abcde'))


    @pytest.mark.parametrize('good_single_regexp',
        (None, '[a-m]*', re.compile('[a-m]+'))
    )
    def test_accepts_good_single(self, good_single_regexp):
        # could be None, str, re.Pattern

        _val_regexp(good_single_regexp, list('abcde'))


    @pytest.mark.parametrize('junk_regexp', ([True, '[a-q]'], ('[1-3]', 0)))
    def test_rejects_junk_regexp_as_seq(self, junk_regexp):
        # could be str, re.Pattern, or list[of the 2]

        with pytest.raises(TypeError):
            _val_regexp(junk_regexp, list('ab'))


    def test_rejects_bad_regexp_as_seq(self):

        # too long
        with pytest.raises(ValueError):
            _val_regexp(['\W' for _ in range(6)], list('abcde'))


        # too short
        with pytest.raises(ValueError):
            _val_regexp(['\W' for _ in range(4)], list('abcde'))


    def test_accepts_good_sequence(self):

        assert _val_regexp(
            [re.compile('a{0, 2}') for _ in range(5)],
            list('abcde')
        ) is None

        assert _val_regexp(
            ['\W' for _ in range(5)],
            list('abcde')
        ) is None

        assert _val_regexp(
            [re.compile('[a-d]*'), re.compile('[a-q]*'), '\W', '\W', '\W'],
            list('abcde')
        ) is None





