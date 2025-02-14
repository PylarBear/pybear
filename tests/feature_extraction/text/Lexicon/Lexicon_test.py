# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text import Lexicon



class TestLexicon:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def Lexicon_instance(scope='module'):
        return Lexicon()

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_find_duplicates_returns_empty_ndarray(self, Lexicon_instance):

        out = Lexicon_instance.find_duplicates()

        assert isinstance(out, np.ndarray)
        assert len(out) == 0


    def test_check_order_returns_empty_ndarray(self, Lexicon_instance):

        out = Lexicon_instance.check_order()

        assert isinstance(out, np.ndarray)
        assert len(out) == 0


    def test_lexicon(self, Lexicon_instance):

        out = Lexicon_instance.lexicon()

        assert isinstance(out, np.ndarray)

        assert len(out) == Lexicon_instance.size


    def _old_py_lexicon(self, Lexicon_instance):

        out = Lexicon_instance._old_py_lexicon()

        assert isinstance(out, np.ndarray)

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



    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # v v v v these are all handled by TextStatistics as of 25_02 v v v v

    def test_size_returns_int(self, Lexicon_instance):

        assert isinstance(Lexicon_instance.size, int)


    def test_lookup_substring(self, Lexicon_instance):

        out = Lexicon_instance.lookup_substring('aard', bypass_validation=False)

        exp = ["AARDVARK", "AARDVARKS", "AARDWOLF", "AARDWOLVES"]

        assert isinstance(out, np.ndarray)
        assert np.array_equiv(out, exp)


        out = Lexicon_instance.lookup_substring('pxlq', bypass_validation=False)

        assert isinstance(out, np.ndarray)
        assert np.array_equiv(out, [])


    def test_lookup_word(self, Lexicon_instance):

        out = Lexicon_instance.lookup_word('tomatoes', bypass_validation=False)

        assert isinstance(out, bool)
        assert out is True


        out = Lexicon_instance.lookup_word('pxlq', bypass_validation=False)

        assert isinstance(out, bool)
        assert out is False


    def test_statistics(self, Lexicon_instance):

        # only prints to screen

        Lexicon_instance.statistics()








































