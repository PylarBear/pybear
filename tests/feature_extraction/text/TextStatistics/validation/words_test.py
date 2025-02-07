# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextStatistics._validation._words import \
    _val_words



class TestStatistics:


    @pytest.mark.parametrize('non_iterable',
        (0, 3.14, True, False, None, lambda x: x, 'garbage', {'a': 1})
    )
    def test_rejects_non_list_like(self, non_iterable):
        with pytest.raises(TypeError):
            _val_words(non_iterable)


    @pytest.mark.parametrize('junk_value',
        (0, 3.14, True, False, None, lambda x: x, {'a': 1}, [1,2])
    )
    def test_rejects_vector_of_non_words(self, junk_value):
        with pytest.raises(TypeError):
            _val_words(['good', 'bad', 'indifferent', junk_value])


    def test_rejects_strings_too_long(self):

        WORDS = [
            'this is a string that is too long to be one word',
            'this is another that is way too long',
            'and yet another that could not be one word'
        ]

        with pytest.raises(ValueError):
            _val_words(WORDS)


    def test_accepts_good_list_of_strs(self):
        _val_words(['good', 'bad', 'indifferent', 'garbage'])


    def test_rejects_empty(self):

        with pytest.raises(ValueError):
            _val_words([])


    def test_handles_small_lists(self):
        _val_words(['short', 'list'])

        _val_words(['alone'])


    def test_catches_junk_characters_in_bucket_called_other(self):

        _val_words(['@junk_characters.com', 'please ignore me!!!!'])




