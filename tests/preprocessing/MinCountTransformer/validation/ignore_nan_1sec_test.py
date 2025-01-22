# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _ignore_nan import _val_ignore_nan

import pytest



@pytest.mark.parametrize('_ignore_nan',
    (0, 1, 3.14, None, 'junk', min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
)
def test_rejects_non_bool(_ignore_nan):
    with pytest.raises(TypeError):
        _val_ignore_nan(_ignore_nan)


@pytest.mark.parametrize('_ignore_nan', (True, False))
def test_accepts_bool(_ignore_nan):
    assert _val_ignore_nan(_ignore_nan) is None









