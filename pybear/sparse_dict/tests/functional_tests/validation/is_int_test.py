# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np
from pybear.sparse_dict import _validation as val



class TestIsInt:
    @pytest.mark.parametrize('x', (-2,-1,0,1,2))
    def test_accepts_integers(self, x):
        val._is_int(x)

    @pytest.mark.parametrize(
        'x',
        (np.pi, float('inf'), [], int, 'junk', {}, True, (1,), None)
    )
    def test_rejects_non_integers(self, x):
        with pytest.raises(TypeError):
            val._is_int((x))


























