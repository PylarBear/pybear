# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextSplitter._validation._validation import \
    _validation



class TestValidation:

    # the brunt of validation is handled in the individual module


    def test_blocks_sep_regexp_both_not_None(self):

        with pytest.raises(ValueError):
            _validation(
                list('abcde'),
                f'[a-b]*',
                {' ', ', ', '. '},
                -1,
                None
            )


    def test_sep_regexp_both_None_must_pass_maxsplit(self):

        with pytest.raises(ValueError):
            _validation(
                list('abcde'),
                None,
                None,
                None,
                None
            )


    def test_sep_passed_must_pass_maxsplit(self):

        with pytest.raises(ValueError):
            _validation(
                list('abcde'),
                None,
                ' ',
                None,
                None
            )


    def test_regexp_passed_do_not_pass_maxsplit(self):

        with pytest.raises(ValueError):
            _validation(
                list('abcde'),
                '[a-q][abcde][123]',
                None,
                -1,
                None
            )


    @pytest.mark.parametrize('regexp',
         ('\W', re.compile('\W'), [re.compile('\d'), '[a-m]*'], None)
    )
    @pytest.mark.parametrize('sep', (' ', {' ', ', '}, [' ', ', '], None))
    @pytest.mark.parametrize('maxsplit', (-1, [1,2], None))
    def test_accepts_good(self, regexp, sep, maxsplit):

        _X = list('ab')

        _will_raise = False

        if sep is not None:
            if maxsplit is None:
                _will_raise += 1
            if regexp is not None:
                _will_raise += 1

        if sep is None:
            if regexp is not None:
                if maxsplit is not None:
                    _will_raise += 1
            if regexp is None:
                if maxsplit is None:
                    _will_raise += 1

        if _will_raise:
            with pytest.raises(ValueError):
                _validation(list('ab'), regexp, sep, maxsplit, None)
        else:
            assert _validation(list('ab'), regexp, sep, maxsplit, None) is None




