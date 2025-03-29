# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextRemover._validation._remove_empty_rows \
    import _val_remove_empty_rows

import pytest



class TestRemoveEmptyRows:


    @pytest.mark.parametrize('junk_remove_empty_rows',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_remove_empty_rows):

        with pytest.raises(TypeError):
            _val_remove_empty_rows(junk_remove_empty_rows)


    @pytest.mark.parametrize('_remove_empty_rows', (True, False) )
    def test_rejects_junk(self, _remove_empty_rows):

        assert _val_remove_empty_rows(_remove_empty_rows) is None





