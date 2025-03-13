# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifierRegExp._validation._line_break \
    import _val_line_break

import pytest

import re



class TestValLinebreak:


    @pytest.mark.parametrize('junk_line_break',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_line_break(self, junk_line_break):
        # must be Union[
        with pytest.raises(TypeError):
            _val_line_break(junk_line_break)


    def test_good_line_break(self):

        assert _val_line_break(None) is None

        assert _val_line_break('[.,;]') is None

        assert _val_line_break(re.compile('[.,;]', flags=re.I)) is None






