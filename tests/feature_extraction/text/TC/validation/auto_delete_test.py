# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TC._validation._auto_add import _val_auto_add



class TestValAutoDelete:

    @pytest.mark.parametrize('junk_auto_delete',
        (-2.7, -1, 0, 1, 2.7, 'garbage', [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_auto_add(self, junk_auto_delete):
        with pytest.raises(TypeError):
            _val_auto_add(junk_auto_delete)


    @pytest.mark.parametrize('auto_delete', (True, False, None))
    def test_accepts_bool_None_auto_delete(self, auto_delete):
        assert _val_auto_add(auto_delete) is None













