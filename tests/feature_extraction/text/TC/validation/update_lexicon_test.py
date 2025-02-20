# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TC._validation._update_lexicon import \
    _val_update_lexicon



class TestValAutoAdd:

    @pytest.mark.parametrize('junk_update_lexicon',
        (-2.7, -1, 0, 1, 2.7, 'garbage', [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_update_lexicon(self, junk_update_lexicon):
        with pytest.raises(TypeError):
            _val_update_lexicon(junk_update_lexicon)


    @pytest.mark.parametrize('update_lexicon', (True, False, None))
    def test_accepts_bool_None_update_lexicon(self, update_lexicon):
        assert _val_update_lexicon(update_lexicon) is None













