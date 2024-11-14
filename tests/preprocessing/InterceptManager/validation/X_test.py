# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation._X \
    import _val_X


import pytest


def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)





