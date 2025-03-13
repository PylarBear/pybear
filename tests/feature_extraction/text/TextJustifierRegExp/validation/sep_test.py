# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifierRegExp._validation._sep import \
    _val_sep

import pytest

import re



class TestValSep:


    @pytest.mark.parametrize('junk_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, None, [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_sep(self, junk_sep):
        with pytest.raises(TypeError):
            _val_sep(junk_sep)


    def test_good_sep(self, ):

        assert _val_sep('[a-d]') is None

        assert _val_sep(re.compile('[a-d]', flags=re.I)) is None







