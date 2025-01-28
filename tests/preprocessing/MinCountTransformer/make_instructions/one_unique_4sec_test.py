# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._make_instructions. \
    _one_unique import _one_unique

import numpy as np

import pytest



class TestOneUnique:

    @pytest.mark.parametrize('_threshold', (3, 5, 10))
    @pytest.mark.parametrize('_nan_key', ('NAN', np.nan, 'nan', False))
    @pytest.mark.parametrize('_nan_ct', (2, 4, 7, False))
    @pytest.mark.parametrize('_key', ('a', 1, 3.14))
    @pytest.mark.parametrize('_value', (2, 5, 9))
    def test_adds_column_and_nan_correctly(self,
        _threshold, _nan_key, _nan_ct, _key, _value):

        if (_nan_key is False) + (_nan_ct is False) == 1:
            pytest.skip(reason=f"disallowed condition")


        out = _one_unique(
                _threshold=_threshold,
                _nan_key=_nan_key,
                _nan_ct=_nan_ct,
                _COLUMN_UNQ_CT_DICT={_key: _value}
        )

        # doesnt matter what the non-nan ct is, rows wont be deleted, only
        # column gets deleted

        # 24_06_11_14_38_00 originally had that nan_ct < thresh and not ignore
        # nan, then delete rows for the nans; changing that to just delete the
        # column
        # if _nan_ct is not False and _nan_ct < _threshold:
        #     assert out == [_nan_key, 'DELETE COLUMN']
        # else:
        assert out == ['DELETE COLUMN']

































