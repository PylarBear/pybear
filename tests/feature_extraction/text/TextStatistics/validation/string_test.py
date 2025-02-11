# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextStatistics._validation._strings import \
    _val_strings



class TestStatistics:


    @pytest.mark.parametrize('non_iterable',
        (0, 3.14, True, False, None, lambda x: x, 'garbage', {'a': 1})
    )
    def test_rejects_non_list_like(self, non_iterable):
        with pytest.raises(TypeError):
            _val_strings(non_iterable)


    @pytest.mark.parametrize('junk_value',
        (0, 3.14, True, False, None, lambda x: x, {'a': 1}, [1,2])
    )
    def test_rejects_vector_of_non_strings(self, junk_value):
        with pytest.raises(TypeError):
            _val_strings(['good', 'bad', 'indifferent', junk_value])


    def test_rejects_empty(self):

        with pytest.raises(ValueError):
            _val_strings([])


    def test_rejects_2D(self):

        with pytest.raises(TypeError):
            _val_strings([['a', 'b', 'c']])


    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    def test_accepts_good_sequence_of_strs(self, container):

        LIST = ['good', 'bad', 'indifferent', 'garbage']

        if container is np.ndarray:
            STRINGS = np.array(LIST)
            assert isinstance(STRINGS, np.ndarray)
        else:
            STRINGS = container(LIST)
            assert isinstance(STRINGS, container)

        _val_strings(STRINGS)


    def test_handles_small_lists(self):
        _val_strings(['short', 'list'])

        _val_strings(['alone'])








