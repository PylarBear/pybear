# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.NoDupPolyFeatures._validation._drop_duplicates import (
    _val_drop_duplicates
)


import pytest



class TestDropDuplicates:


    @pytest.mark.parametrize('junk_drop_duplicates',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_drop_duplicates):
        with pytest.raises(TypeError):
            _val_drop_duplicates(junk_drop_duplicates)


    @pytest.mark.parametrize('good_drop_duplicates', (True, False))
    def test_accepts_bool(self, good_drop_duplicates):
        _val_drop_duplicates(good_drop_duplicates)
