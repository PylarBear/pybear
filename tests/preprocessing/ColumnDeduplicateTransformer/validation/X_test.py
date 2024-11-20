# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation._X \
    import _val_X

import numpy as np
import scipy.sparse as ss

import pytest




def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)



def test_X_must_have_at_least_1_example():

    with pytest.raises(ValueError):
        _val_X(np.random.randint(0, 10, (0, 10)))



def test_rejects_scipy_bsr():


    _X = np.random.randint(0, 10, (10, 2))


    with pytest.raises(TypeError):
        _val_X(ss.bsr_matrix(_X))


    with pytest.raises(TypeError):
        _val_X(ss.bsr_array(_X))







