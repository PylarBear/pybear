# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.feature_extraction.text._TextCleaner._lex_lookup. \
    _lex_lookup_menu import _lex_lookup_menu




class TestLexLookupMenu:

    @staticmethod
    @pytest.fixture
    def good_LLDICT():

        return {
            'A': 'FOO',
            'B': 'BAR',
            'C': 'BAZ',
            'D': 'QUX',
            'E': 'QUUX'
        }


    @pytest.mark.parametrize('junk_allowed',
        (0, 1, 3.14, True, min, [1,2], (1,2), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_allowed(self, good_LLDICT, junk_allowed):

        with pytest.raises(TypeError):

            _lex_lookup_menu(
                good_LLDICT,
                allowed=junk_allowed,
                disallowed=None
            )

    @pytest.mark.parametrize('bad_allowed', ('x', 'y', 'z'))
    def test_rejects_bad_allowed(self, good_LLDICT, bad_allowed):

        with pytest.raises(ValueError):

            _lex_lookup_menu(
                good_LLDICT,
                allowed=bad_allowed,
                disallowed=None
            )


    @pytest.mark.parametrize('junk_disallowed',
        (0, 1, 3.14, True, min, [1,2], (1,2), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_disallowed(self, good_LLDICT, junk_disallowed):

        with pytest.raises(TypeError):

            _lex_lookup_menu(
                good_LLDICT,
                allowed=None,
                disallowed=junk_disallowed
            )


    @pytest.mark.parametrize('bad_disallowed', ('x', 'y', 'z'))
    def test_rejects_bad_disallowed(self, good_LLDICT, bad_disallowed):

        with pytest.raises(ValueError):

            _lex_lookup_menu(
                good_LLDICT,
                allowed=None,
                disallowed=bad_disallowed
            )



    def test_rejects_allowed_and_disallowed(self, good_LLDICT):

        with pytest.raises(ValueError):

            _lex_lookup_menu(
                good_LLDICT,
                allowed='abc',
                disallowed='de'
            )


    def test_all_None_returns_all_allowed(self, good_LLDICT):

        exp_disp = ", ".join([f'{v}({k.lower()})' for k, v in good_LLDICT.items()])
        exp_allowed = "".join(list(good_LLDICT.keys())).lower()

        out_disp, out_allowed = _lex_lookup_menu(
            good_LLDICT,
            allowed=None,
            disallowed=None
        )

        assert out_disp == exp_disp
        assert out_allowed == exp_allowed



    def test_accuracy(self, good_LLDICT):

        out_disp, out_allowed = _lex_lookup_menu(
            good_LLDICT,
            allowed='ace',
            disallowed=None
        )

        exp_disp = ", ".join([f'{v}({k.lower()})' for k, v in good_LLDICT.items() \
                              if k.lower() in 'ace'])

        assert out_disp == exp_disp
        assert out_allowed == "ace"


        out_disp, out_allowed = _lex_lookup_menu(
            good_LLDICT,
            allowed=None,
            disallowed='bde'
        )

        exp_disp = ", ".join([f'{v}({k.lower()})' for k, v in good_LLDICT.items() \
                              if k.lower() in 'ac'])

        assert out_disp == exp_disp
        assert out_allowed == "ac"














































