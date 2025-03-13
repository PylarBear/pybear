# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextJustifierRegExp._validation. \
    _line_break_flags import _val_line_break_flags



class TestValLineBreakFlags:


    @pytest.mark.parametrize('junk_line_break_flags',
        (-2.7, 2.7, True, False, 'trash', [0,1], (0,1), {1,2}, {"a":1}, lambda x: x)
    )
    def test_rejects_non_integers(self, junk_line_break_flags):

        with pytest.raises(TypeError):
            _val_line_break_flags(junk_line_break_flags)


    @pytest.mark.parametrize('line_break_flags', (-1, 0, 2, None))
    def test_accepts_good(self, line_break_flags):

        assert _val_line_break_flags(line_break_flags) is None






