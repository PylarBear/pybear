# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextCleaner._shared. \
    _validation._X import _val_X



class TestValX:


    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {1,2}, {'a':1},
        lambda x: x)
    )
    def test_blocks_junk(self, junk_X):

        with pytest.raises(TypeError):

            _val_X(junk_X)


    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    def test_accepts_good(self, container):

        _strings = list('abcdefg')

        if container is np.ndarray:
            _X = np.array(_strings)
            assert isinstance(_X, np.ndarray)
        else:
            _X = container(_strings)
            assert isinstance(_X, container)

        assert _val_X(_X) is None


