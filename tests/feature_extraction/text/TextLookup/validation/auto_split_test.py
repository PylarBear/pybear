# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._validation._auto_split import \
    _val_auto_split



class TestAutoSplit:


    @pytest.mark.parametrize('junk_auto_split',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_auto_split):

        with pytest.raises(TypeError):
            _val_auto_split(junk_auto_split)


    @pytest.mark.parametrize('_auto_split', (True, False))
    def test_accepts_bool(self, _auto_split):

        assert _val_auto_split(_auto_split) is None








