# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.feature_extraction.text._TC._methods._validation._wip_X import \
    _val_wip_X




class TestValWipX:


    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {'A':1},
         lambda x: x, [[1,2,3], [4,5,6]])
    )
    def test_rejects_junk(self, junk_X):

        with pytest.raises(TypeError):

            _val_wip_X(junk_X)


    @pytest.mark.parametrize('container',  (list, tuple, set, np.ndarray))
    def test_accepts_good_1D(self, container):

        _words = ['I', 'do', 'not', 'like', 'green', 'eggs', 'and', 'ham']

        if container is np.ndarray:
            _X = np.array(_words)
            assert isinstance(_X, np.ndarray)
        else:
            _X = container(_words)
            assert isinstance(_X, container)

        _val_wip_X(_X)


    @pytest.mark.parametrize('container',  (list, tuple, np.ndarray))
    def test_accepts_good_2D(self, container):

        _words = [
            ['Sam', 'I', 'do'],
            ['not', 'like', 'green'],
            ['eggs', 'and', 'ham']
        ]

        if container is np.ndarray:
            _X = np.array(_words)
            assert isinstance(_X, np.ndarray)
        else:
            _X = container(map(container, _words))
            assert isinstance(_X, container)

        _val_wip_X(_X)









