# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation._original_dtypes \
    import _val_original_dtypes

import numpy as np

import pytest



class TestValOriginalDtypes:


    @pytest.mark.parametrize('junk_og_dtypes',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_og_dtypes(self, junk_og_dtypes):

        with pytest.raises(TypeError):
            _val_original_dtypes(junk_og_dtypes)


    @pytest.mark.parametrize('junk_dtype',
        (0, True, 3.14, min, None, lambda x: x, {'a':1})
    )
    def test_rejects_junk_values(self, junk_dtype):
        with pytest.raises(TypeError):
            _val_original_dtypes(
                np.array(['int', 'float', 'obj', junk_dtype], dtype=object)
            )


    @pytest.mark.parametrize('container', (list, set, tuple, np.array))
    @pytest.mark.parametrize('bad_og_dtype',
        (list('aaa'), list('bbb'), list('xyz'))
    )
    def test_rejects_bad_values(self, container, bad_og_dtype):

        with pytest.raises(ValueError):
            _val_original_dtypes(container(bad_og_dtype))


    VALUES = ['bin_int', 'obj', 'int', 'float']
    @pytest.mark.parametrize('good_ogdtype',
        (list(VALUES), set(VALUES), tuple(VALUES), np.array(VALUES))
    )
    def test_accepts_good_og_dtype(self, good_ogdtype):

        out = _val_original_dtypes(good_ogdtype)

        assert out is None















