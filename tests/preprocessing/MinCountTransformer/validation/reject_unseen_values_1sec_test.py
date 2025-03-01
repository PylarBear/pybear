# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._MinCountTransformer._validation. \
    _reject_unseen_values import _val_reject_unseen_values

import pytest



@pytest.mark.parametrize('_reject_unseen_values',
    (0, 1, 3.14, None, 'junk', min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
)
def test_rejects_non_bool(_reject_unseen_values):
    with pytest.raises(TypeError):
        _val_reject_unseen_values(_reject_unseen_values)


@pytest.mark.parametrize('_reject_unseen_values', (True, False))
def test_accepts_bool(_reject_unseen_values):
    assert _val_reject_unseen_values(_reject_unseen_values) is None









