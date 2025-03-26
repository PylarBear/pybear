# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._validation._wrap import _val_wrap

import pytest




class TestValWrap:


    @pytest.mark.parametrize('junk_wrap',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_wrap):

        with pytest.raises(TypeError):
            _val_wrap(junk_wrap)


    @pytest.mark.parametrize('_wrap', (True, False))
    def test_rejects_non_bool(self, _wrap):

        assert _val_wrap(_wrap) is None


