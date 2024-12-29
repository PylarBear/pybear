# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing.MinCountTransformer._validation._val_transform import \
    _val_transform



class TestValTransform:

    @pytest.mark.parametrize('_junk',
        (-1, 0, 1, 3.14, True, False, min, [1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_val_transform_rejects_junk(self, _junk):

        with pytest.raises(TypeError):
            _val_transform(_junk)

    @pytest.mark.parametrize('_bad',
         ('junk', 'trash', 'garbage', 'rubbish', 'waste')
    )
    def test_val_transform_rejects_junk(self, _bad):

        with pytest.raises(ValueError):
            _val_transform(_bad)



    ALLOWED_TRANSFORMS = [
        "default",
        "numpy_array",
        "pandas_dataframe",
        "pandas_series",
        None
    ]

    @pytest.mark.parametrize('_good', ALLOWED_TRANSFORMS)
    def test_val_transform_accepts_good(self, _good):

        assert _val_transform(_good) == _good








