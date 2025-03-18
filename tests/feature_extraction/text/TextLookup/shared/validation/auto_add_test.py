# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._shared._validation. \
    _auto_add_to_lexicon import _val_auto_add_to_lexicon



class TestAutoAdd:


    @pytest.mark.parametrize('junk_auto_add_to_lexicon',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_auto_add_to_lexicon):

        with pytest.raises(TypeError):
            _val_auto_add_to_lexicon(junk_auto_add_to_lexicon)


    @pytest.mark.parametrize('_auto_add_to_lexicon', (True, False))
    def test_accepts_bool(self, _auto_add_to_lexicon):

        assert _val_auto_add_to_lexicon(_auto_add_to_lexicon) is None








