# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextReplacer._validation._1D_str_sequence \
    import _val_1D_str_sequence



class TestVal1DStrSequence:


    @pytest.mark.parametrize('junk_vector',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {1,2}, {'a':1},
        lambda x: x)
    )
    def test_blocks_junk(self, junk_vector):

        with pytest.raises(TypeError):

            _val_1D_str_sequence(junk_vector)


    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    def test_accepts_good(self, container):

        _strings = list('abcdefg')

        if container is np.ndarray:
            _vector = np.array(_strings)
            assert isinstance(_vector, np.ndarray)
        else:
            _vector = container(_strings)
            assert isinstance(_vector, container)

        assert _val_1D_str_sequence(_vector) is None








