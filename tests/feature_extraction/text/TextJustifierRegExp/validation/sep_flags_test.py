# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextJustifierRegExp._validation._sep_flags \
    import _val_sep_flags



class TestValSepFlags:


    @pytest.mark.parametrize('junk_sep_flags',
        (-2.7, 2.7, True, False, 'trash', [0,1], (0,1), {1,2}, {"a":1}, lambda x: x)
    )
    def test_rejects_non_integers(self, junk_sep_flags):

        with pytest.raises(TypeError):
            _val_sep_flags(junk_sep_flags)


    @pytest.mark.parametrize('sep_flags', (-1, 0, 2, None))
    def test_accepts_good(self, sep_flags):

        assert _val_sep_flags(sep_flags) is None






