# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextCleaner._shared. \
    _validation._2D_str_sequence import _val_2D_str_array



class TestVal2DStrArray:


    @pytest.mark.parametrize('junk_array',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {1,2}, {'a':1},
        lambda x: x, ({'a':1}, {'b':2}), [[1,2], [3,4]], [['a', 'b'], [1, 'c']])
    )
    def test_blocks_junk(self, junk_array):

        with pytest.raises(TypeError):

            _val_2D_str_array(junk_array)


    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    @pytest.mark.parametrize('ragged', (True, False))
    def test_accepts_good(self, container, ragged):

        _strings = list('abcdefg')

        if container is np.ndarray:
            if ragged:
                pytest.skip(reason=f'cant make ragged np array')
            else:
                _array = np.repeat(_strings, 3).reshape(3, 7)
                assert isinstance(_array, np.ndarray)
                assert _array.shape == (3, 7)
        else:
            if container is set:
                pytest.skip(reason=f'cant make sets of sets')
            _array = container(container(_strings) for i in range(3))
            assert isinstance(_array, container)


        assert _val_2D_str_array(_array) is None








