# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TC._validation._X import _val_X




class TestValX:



    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [['a', 'b']], {'a':1}, lambda x: x)
    )
    def test_rejects_junk_X(self, junk_X):

        with pytest.raises(TypeError):
            _val_X(junk_X)



    @pytest.mark.parametrize('container', (list, set, tuple, np.ndarray))
    def test_accepts_1D_seq_of_str(self, container):

        _strs = list('abcdefg')

        if container is np.ndarray:
            X = np.array(_strs)
            assert isinstance(X, np.ndarray)
        else:
            X = container(_strs)
            assert isinstance(X, container)

        _val_X(X)





