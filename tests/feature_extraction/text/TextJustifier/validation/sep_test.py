# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._validation._sep import \
    _val_sep

import pytest



class TestValSep:


    @pytest.mark.parametrize('junk_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, None, [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_sep(self, junk_sep):
        with pytest.raises(TypeError):
            _val_sep(junk_sep)


    def test_rejects_empty_string(self):

        with pytest.raises(ValueError):
            _val_sep('')

        with pytest.raises(ValueError):
            _val_sep({'', ' ', '_'})


    def test_rejects_empty_set(self):

        with pytest.raises(ValueError):
            _val_sep(set())


    @pytest.mark.parametrize('good_sep',
        ('priceless', 'invaluable', 'precious', set('abc'), set('123'))
    )
    def test_good_sep(self, good_sep):

        assert _val_sep(good_sep) is None



