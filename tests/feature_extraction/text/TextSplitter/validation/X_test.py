# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter._validation._X import _val_X



class TestValX:


    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1, ), {'A':1},
         lambda x: x, np.random.randint(0, 10, (5,3)), np.random.randint(0, 10, (5,)))
    )
    def test_rejects_junk_X(self, junk_X):

        with pytest.raises(TypeError):
            _val_X(junk_X)



    def test_accepts_1D_of_str(self):

        _val_X(list('abcde'))

        _val_X(set('abcde'))

        _val_X(tuple('abcde'))

        _val_X(np.array(list('abcde')))

