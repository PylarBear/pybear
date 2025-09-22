# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextLookup._shared._validation. \
    _replace_always import _val_replace_always



class TestReplaceAlways:


    def test_accepts_None(self):

        assert _val_replace_always(None) is None


    def test_rejects_empty(self):

        # empty dict
        with pytest.raises(ValueError):
            _val_replace_always({})


    @pytest.mark.parametrize('key',
        (0, 2.7, True, False, 'trash', re.compile('^a+$', re.I), [0,1], lambda x: x)
    )
    @pytest.mark.parametrize('value',
        (0, 2.7, True, False, 'trash', re.compile('^a+$', re.I), [0,1], lambda x: x)
    )
    def test_accepts_good(self, key, value):

        # must be dict[str | re.Pattern[str], str]

        try:
            {key: value}
        except:
            pytest.skip(reason=f"cant do a test if cant make a dict")


        if isinstance(key, (str, re.Pattern)) and isinstance(value, str):
            assert _val_replace_always({key: value}) is None
        else:
            with pytest.raises(TypeError):
                assert _val_replace_always({key: value}) is None







