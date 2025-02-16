# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza see if we need any of this stuff


import pytest

import numpy as np

from pybear.feature_extraction.text import Lexicon

pytest.skip(reason=f"pizza say so", allow_module_level=True)

class TestLexicon:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def Lexicon_instance(scope='module'):
        return Lexicon()

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *





    def _old_py_lexicon(self, Lexicon_instance):

        out = Lexicon_instance._old_py_lexicon()

        assert isinstance(out, list)

        # as of 24_06_21 this is passing, but because the files and array
        # containers are independent, the arrays could fall into neglect
        # and make this not true.

        assert len(out) == Lexicon_instance.size()


    @pytest.mark.parametrize('junk', (0, np.pi, 'trash', [1,2], (1,2), {1,2}))
    def test_validation_of_bypass_validation_junk(self, junk, Lexicon_instance):

        # lookup_substring
        with pytest.raises(ValueError):
            Lexicon_instance.lookup_substring('aard', bypass_validation=junk)

        # lookup_word
        with pytest.raises(ValueError):
            Lexicon_instance.lookup_word('aard', bypass_validation=junk)


    @pytest.mark.parametrize('good', (True, False, None))
    def test_validation_of_bypass_validation_good(self, good, Lexicon_instance):

        # lookup_substring
        Lexicon_instance.lookup_substring('aard', bypass_validation=good)

        # lookup_word
        Lexicon_instance.lookup_word('aard', bypass_validation=good)










