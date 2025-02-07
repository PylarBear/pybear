# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
from unittest.mock import patch

import io

from pybear.feature_extraction.text._TextCleaner._lex_lookup. \
    _display_lexicon_update import _display_lexicon_update



class TestDisplayLexiconUpdate:

    @staticmethod
    @pytest.fixture
    def addendum():
        return [
            "AGORAPHOBIA",
            "RHUBARB",
            "QUIZZICAL",
            "BYZANTINE"
        ]



    def test_displays_addendum(self, addendum):

        user_inputs = "\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            out = _display_lexicon_update(addendum)

        assert out is None










