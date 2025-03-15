# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._validation._update_lexicon \
    import _val_update_lexicon



class TestUpdateLexicon:


    @pytest.mark.parametrize('junk_update_lexicon',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_update_lexicon):

        with pytest.raises(TypeError):
            _val_update_lexicon(junk_update_lexicon)


    @pytest.mark.parametrize('_update_lexicon', (True, False))
    def test_accepts_bool(self, _update_lexicon):

        assert _val_update_lexicon(_update_lexicon) is None








