# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._validation._sep import _val_sep

import pytest



class TestValSep:


    @pytest.mark.parametrize('junk_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_sep):

        with pytest.raises(TypeError):
            _val_sep(junk_sep)


    def test_accepts_str_None(self):

        assert _val_sep('any old string') is None

        assert _val_sep(None) is None








