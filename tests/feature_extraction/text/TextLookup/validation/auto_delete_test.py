# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._validation._auto_delete import \
    _val_auto_delete



class TestAutoDelete:


    @pytest.mark.parametrize('junk_auto_delete',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_auto_delete):

        with pytest.raises(TypeError):
            _val_auto_delete(junk_auto_delete)


    @pytest.mark.parametrize('_auto_delete', (True, False))
    def test_accepts_bool(self, _auto_delete):

        assert _val_auto_delete(_auto_delete) is None








