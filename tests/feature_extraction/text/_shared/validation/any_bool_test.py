# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text.__shared._validation._any_bool \
    import _val_any_bool

import pytest



class TestValAnyBool:


    @pytest.mark.parametrize('junk_any_bool',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_any_bool):

        with pytest.raises(TypeError):
            _val_any_bool(junk_any_bool)


    @pytest.mark.parametrize('_any_bool', (True, False) )
    def test_rejects_junk(self, _any_bool):

        assert _val_any_bool(_any_bool) is None





