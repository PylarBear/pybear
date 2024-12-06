# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.NoDupPolyFeatures._validation._drop_collinear import (
    _val_drop_collinear
)


import pytest



class TestDropCollinear:


    @pytest.mark.parametrize('junk_drop_collinear',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_drop_collinear):
        with pytest.raises(TypeError):
            _val_drop_collinear(junk_drop_collinear)


    @pytest.mark.parametrize('junk_drop_collinear', (True, False))
    def test_accepts_bool(self, junk_drop_collinear):
        _val_drop_collinear(junk_drop_collinear)
