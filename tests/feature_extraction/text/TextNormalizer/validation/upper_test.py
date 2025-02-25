# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextNormalizer._validation._upper import \
    _val_upper




class TestValUpper:


    @pytest.mark.parametrize('junk_upper',
        (-2.7, -1, 0, 1, 2.7, 'trash', [0,1], (1,), {1,2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_not_None_bool(self, junk_upper):

        with pytest.raises(TypeError):
            _val_upper(junk_upper)




    @pytest.mark.parametrize('_upper', (True, False, None))
    def test_accepts_None_bool(self, _upper):

        assert _val_upper(_upper) is None






