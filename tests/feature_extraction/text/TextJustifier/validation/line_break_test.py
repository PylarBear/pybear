# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._validation._line_break \
    import _val_line_break

import pytest



class TestValLinebreak:


    @pytest.mark.parametrize('junk_line_break',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_line_break(self, junk_line_break):
        # must be Union[
        with pytest.raises(TypeError):
            _val_line_break(junk_line_break)


    def test_rejects_empty_str(self):

        with pytest.raises(ValueError):
            _val_line_break('')

        with pytest.raises(ValueError):
            _val_line_break({'', ' ', '_'})


    def test_rejects_empty_set(self):

        with pytest.raises(ValueError):
            _val_line_break(set())


    @pytest.mark.parametrize('good_line_break',
        ('priceless', 'invaluable', 'precious', set('abc'), set('123'), None)
    )
    def test_good_line_break(self, good_line_break):

        assert _val_line_break(good_line_break) is None





