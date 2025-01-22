# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._shared._make_instructions. \
    _validation._original_dtypes import _val_original_dtypes

import numpy as np

import pytest



class TestValOriginalDtypes:

    @pytest.mark.parametrize('junk_og_dtype',
        (0, True, 3.14, min, None, lambda x: x)
    )
    def test_rejects_any_non_iterable(self, junk_og_dtype):

        with pytest.raises(TypeError):
            _val_original_dtypes(junk_og_dtype)


    @pytest.mark.parametrize('bad_iterable',
        ('junk', [1,2,3], {1,2,3}, (1,2,3), {'a':1})
    )
    def test_rejects_any_iterable_not_a_ndarray(self, bad_iterable):

        with pytest.raises(TypeError):
            _val_original_dtypes(bad_iterable)


    def test_passes_good_ndarray(self):
        out = _val_original_dtypes(np.array(['int', 'float', 'int', 'obj']))
        assert out is None


    @pytest.mark.parametrize('junk_dtype',
        (0, True, 3.14, min, None, lambda x: x, [1,2], (1,2), {1,2}, {'a':1})
    )
    def test_rejects_bad_values(self, junk_dtype):
        with pytest.raises(ValueError):
            _val_original_dtypes(np.array(['int', 'float', 'obj', junk_dtype]))


    @pytest.mark.parametrize('bad_dtype', ('aaa', 'bbb', 'xyz', 'integer'))
    def test_rejects_bad_values(self, bad_dtype):

        with pytest.raises(ValueError):
            _val_original_dtypes(np.array(['int', 'float', 'obj', bad_dtype]))













