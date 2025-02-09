# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

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

    @pytest.mark.xfail(reason=f"pizza needs to decide about phrases or words")
    def test_rejects_spaces(self):

        WORDS = [
            'this isnt too long',
            'one space',
            'sixty six'
        ]

        with pytest.raises(ValueError):
            _val_words(WORDS)

    @pytest.mark.xfail(reason=f"pizza needs to decide about phrases or words")
    def test_rejects_strings_too_long(self):

        WORDS = [
            'floccinaucinihilipilificationisacceptedasaword',
            'reallyneedtosplitthisstringintoindividualwords',
            'usepybearTextCleanertopreprocessesyourstrings'
        ]

        with pytest.raises(ValueError):
            _val_words(WORDS)


    def test_rejects_empty(self):

        with pytest.raises(ValueError):
            _val_words([])


    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    def test_accepts_good_sequence_of_strs(self, container):

        LIST = ['good', 'bad', 'indifferent', 'garbage']

        if container is np.ndarray:
            WORDS = np.array(LIST)
            assert isinstance(WORDS, np.ndarray)
        else:
            WORDS = container(LIST)
            assert isinstance(WORDS, container)

        _val_words(WORDS)


    def test_handles_small_lists(self):
        _val_words(['short', 'list'])

        _val_words(['alone'])








