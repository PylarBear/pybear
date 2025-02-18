# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextStatistics._validation._store_uniques \
    import _val_store_uniques



class TestValStoreUniques:


    @pytest.mark.parametrize('junk_su',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_su):

        with pytest.raises(TypeError):
            _val_store_uniques(junk_su)


    @pytest.mark.parametrize('_su', (True, False) )
    def test_accepts_bool(self, _su):

        _val_store_uniques(_su)








