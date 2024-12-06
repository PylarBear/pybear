# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.NoDupPolyFeatures._validation._drop_constants import (
    _val_drop_constants
)


import pytest



class TestDropConstants:


    @pytest.mark.parametrize('junk_drop_constants',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_drop_constants):
        with pytest.raises(TypeError):
            _val_drop_constants(junk_drop_constants)


    @pytest.mark.parametrize('good_drop_constants', (True, False))
    def test_accepts_bool(self, good_drop_constants):
        _val_drop_constants(good_drop_constants)

