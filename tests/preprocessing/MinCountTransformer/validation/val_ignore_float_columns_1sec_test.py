# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _val_ignore_float_columns import _val_ignore_float_columns

import pytest



@pytest.mark.parametrize('_ignore_float_columns',
    (0, 1, 3.14, None, 'junk', min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
)
def test_rejects_non_bool(_ignore_float_columns):
    with pytest.raises(TypeError):
        _val_ignore_float_columns(_ignore_float_columns)


@pytest.mark.parametrize('_ignore_float_columns', (True, False))
def test_accepts_bool(_ignore_float_columns):
    assert _val_ignore_float_columns(_ignore_float_columns) is \
        _ignore_float_columns









