# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextLookup._shared._validation._skip_numbers \
    import _val_skip_numbers



class TestSkipNumbers:


    @pytest.mark.parametrize('junk_skip_numbers',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, junk_skip_numbers):

        with pytest.raises(TypeError):
            _val_skip_numbers(junk_skip_numbers)


    @pytest.mark.parametrize('_skip_numbers', (True, False))
    def test_accepts_bool(self, _skip_numbers):

        assert _val_skip_numbers(_skip_numbers) is None








