# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from pybear.preprocessing.MinCountTransformer._shared._validation. \
    _val_ignore_non_binary_integer_columns import \
    _val_ignore_non_binary_integer_columns




@pytest.mark.parametrize('_inbic',
    (0, 1, 3.14, None, 'junk', min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
)
def test_rejects_non_bool(_inbic):
    with pytest.raises(TypeError):
        _val_ignore_non_binary_integer_columns(_inbic)


@pytest.mark.parametrize('_inbic', (True, False))
def test_accepts_bool(_inbic):
    assert _val_ignore_non_binary_integer_columns(_inbic) is _inbic









