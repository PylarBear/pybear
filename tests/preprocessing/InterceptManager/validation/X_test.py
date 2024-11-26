# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._validation._X import _val_X

import numpy as np

import pytest


def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)


def test_X_must_have_1_example():

    with pytest.raises(ValueError):
        _val_X(np.empty((0, 10), dtype=np.float64))


    _val_X(np.random.uniform(0, 10, (100, 10)))









