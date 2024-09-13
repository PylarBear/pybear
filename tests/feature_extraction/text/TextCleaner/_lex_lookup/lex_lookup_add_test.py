# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from copy import deepcopy
from pybear.feature_extraction.text._TextCleaner._lex_lookup. \
    _lex_lookup_add import _lex_lookup_add





class TestLexLookupAdd:

    @staticmethod
    @pytest.fixture
    def good_LA():
        return [
            'ANIMAL',
            'KAKISTOCRACY',
            'FISH',
            'EGGS'
        ]

    @staticmethod
    @pytest.fixture
    def good_KW():
        return [
            'KAKISTOCRACY',
            'FISH',
            'EGGS',
            'ANIMAL'
        ]


    @pytest.mark.parametrize('bad_word',
        (0, True, 3.14, None, min, [0,1], (0,1), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_word_not_str(self, bad_word, good_LA, good_KW):

        with pytest.raises(TypeError):
            _lex_lookup_add(bad_word, good_LA, good_KW)


    @pytest.mark.parametrize('bad_LA',
        (0, True, 3.14, None, 'junk', min, [0,1], (0,1), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_LA_not_list_of_str(self, bad_LA, good_KW):

        with pytest.raises(TypeError):
            _lex_lookup_add('whatever', bad_LA, deepcopy(good_KW))


    @pytest.mark.parametrize('bad_KW',
        (0, True, 3.14, None, 'junk', min, [0,1], (0,1), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_KW_not_list_of_str(self, good_LA, bad_KW):

        with pytest.raises(TypeError):
            _lex_lookup_add('whatever', deepcopy(good_LA), bad_KW)



    @pytest.mark.parametrize('good_word',
        ('baseball', 'FoOtBaLl', 'hockey', 'basketball', 'CURLING')
    )
    def test_accuracy(self, good_word, good_LA, good_KW):

        OUT_LA, OUT_KW = _lex_lookup_add(good_word, good_LA, good_KW)

        EXP_LA = deepcopy(good_LA)
        EXP_LA.append(good_word.upper())
        EXP_LA.sort()

        EXP_KW = deepcopy(good_KW)
        EXP_KW.append(good_word.upper())
        EXP_KW.sort()

        assert OUT_LA == EXP_LA
        assert OUT_KW == EXP_KW












