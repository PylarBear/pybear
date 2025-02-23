# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextPadder._validation._fill import \
    _val_fill



class TestValFill:


    @pytest.mark.parametrize(f'junk_fill',
        (-2.7, -1, 0, 1, 2.7, True, None, [0,1], (1,), {1,2}, {"A":1}, lambda x: x)
    )
    def test_rejects_non_str(self, junk_fill):

        with pytest.raises(TypeError):
            _val_fill(junk_fill)



    def test_accepts_string(self):

        assert _val_fill('') is None






