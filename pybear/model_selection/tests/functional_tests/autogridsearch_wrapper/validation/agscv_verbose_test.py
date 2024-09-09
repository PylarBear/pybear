# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation.\
    _agscv_verbose import _agscv_verbose


class TestAgscvVerbose:


    @pytest.mark.parametrize('non_null',
        (1, 3.14, True, 'string', (1,2), {1,2}, {'a':1}, lambda x: x)
    )
    def test_accepts_any_non_null_and_returns_bool(self, non_null):
        assert _agscv_verbose(non_null) is True


    @pytest.mark.parametrize('null',
        (0, None, False, [], (), {})
    )
    def test_any_null_returns_false(self, null):
        assert _agscv_verbose(null) is False








