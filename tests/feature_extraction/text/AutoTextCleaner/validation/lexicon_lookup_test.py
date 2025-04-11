# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._AutoTextCleaner._validation._lexicon_lookup \
    import _val_lexicon_lookup

import pytest



class TestValLexiconLookup:


    @pytest.mark.parametrize('_junk_ll',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, _junk_ll):

        with pytest.raises(TypeError):

            _val_lexicon_lookup(_junk_ll)


    @pytest.mark.parametrize('_bad_ll', ('trash', 'garbage', 'before', 'after'))
    def test_rejects_bad(self, _bad_ll):

        with pytest.raises(ValueError):

            _val_lexicon_lookup(_bad_ll)


    @pytest.mark.parametrize('_ll', (None, 'auto_add', 'auto_delete', 'manual'))
    def test_accepts_good(self, _ll):

        assert _val_lexicon_lookup(_ll) is None

















