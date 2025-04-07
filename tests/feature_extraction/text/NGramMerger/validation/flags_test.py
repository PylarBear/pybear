# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._validation._flags import \
    _val_flags

import pytest

import re



class TestValFlags:


    # must be None or numbers.Integral


    @pytest.mark.parametrize('junk_flags',
        (-2.7, 2.7, True, False, 'garbage', (0,1), {1,2}, {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_flags(self, junk_flags):

        with pytest.raises(TypeError):
            _val_flags(junk_flags)


    def test_accepts_None_int(self):

        assert _val_flags(None) is None

        assert _val_flags(-20) is None

        assert _val_flags(10_000) is None

        assert _val_flags(0) is None

        assert _val_flags(re.I | re.X) is None








