# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.NoDupPolyFeatures._validation. \
    _scan_X import _val_scan_X


import pytest



class TestIgnoreNan:


    @pytest.mark.parametrize('junk_ign_nan',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_ign_nan):
        with pytest.raises(TypeError):
            _val_scan_X(junk_ign_nan)


    @pytest.mark.parametrize('good_ign_nan', (True, False))
    def test_accepts_bool(self, good_ign_nan):
        _val_scan_X(good_ign_nan)



    def test_warns_when_false(self):
        with pytest.warns():
            _val_scan_X(False)




