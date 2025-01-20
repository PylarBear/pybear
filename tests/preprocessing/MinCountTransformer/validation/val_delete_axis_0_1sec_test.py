# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _val_delete_axis_0 import _val_delete_axis_0

import pytest



@pytest.mark.parametrize('_delete_axis_0',
    (0, 1, 3.14, None, 'junk', min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
)
def test_rejects_non_bool(_delete_axis_0):
    with pytest.raises(TypeError):
        _val_delete_axis_0(_delete_axis_0)


@pytest.mark.parametrize('_delete_axis_0', (True, False))
def test_accepts_bool(_delete_axis_0):
    assert _val_delete_axis_0(_delete_axis_0) is _delete_axis_0









