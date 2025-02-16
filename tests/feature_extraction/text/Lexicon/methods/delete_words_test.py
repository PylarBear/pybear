# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._Lexicon._methods._delete_words import \
    _delete_words



class TestDeleteWords:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_WORDS',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_WORDS(self, junk_WORDS):
        with pytest.raises(TypeError):
            _delete_words(
                junk_WORDS,
                lexicon_folder_path='sam i am',
                case_sensitive=False
            )

    @pytest.mark.parametrize('junk_path',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_lexicon_folder_path(self, junk_path):
        with pytest.raises(TypeError):
            _delete_words(
                'ULTRACREPIDARIAN',
                lexicon_folder_path=junk_path,
                case_sensitive=False
            )


    @pytest.mark.parametrize('junk_cs',
        (-2.7, -1, 0, 1, 2.7, None, [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_case_sensitive(self, junk_cs):
        with pytest.raises(TypeError):
            _delete_words(
                'CREPUSCULAR',
                lexicon_folder_path='/somewhere/out/there',
                case_sensitive=junk_cs
            )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



