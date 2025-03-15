# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._validation._verbose import \
    _val_verbose



class TestVerbose:


    @pytest.mark.parametrize('junk_verbose',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_verbose):

        with pytest.raises(TypeError):
            _val_verbose(junk_verbose)


    @pytest.mark.parametrize('_verbose', (True, False))
    def test_accepts_bool(self, _verbose):

        assert _val_verbose(_verbose) is None








