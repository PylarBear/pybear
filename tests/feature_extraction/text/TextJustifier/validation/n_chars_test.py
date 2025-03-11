# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest


from pybear.feature_extraction.text._TextJustifier._validation._n_chars import \
    _val_n_chars




class TestValNChar:


    @pytest.mark.parametrize('junk_n_char',
        (-2.7, 2.7, True, False, None, 'trash', [0,1], (1,), {1,2}, {'A':1},
         lambda x: x)
    )
    def test_n_char_rejects_junk(self, junk_n_char):

        with pytest.raises(TypeError):
            _val_n_chars(junk_n_char)


    @pytest.mark.parametrize('bad_n_char', (-100, -2, -1, 0))
    def test_n_char_rejects_bad(self, bad_n_char):
        with pytest.raises(ValueError):
            _val_n_chars(bad_n_char)


    @pytest.mark.parametrize('good_n_char', (1, 2, 3, 100))
    def test_n_char_accepts_good(self, good_n_char):

        assert _val_n_chars(good_n_char) is None







