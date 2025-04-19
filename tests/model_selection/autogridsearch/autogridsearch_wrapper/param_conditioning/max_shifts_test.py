# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _param_conditioning._max_shifts import _cond_max_shifts

import pytest



class TestCondMaxShifts:


    @pytest.mark.parametrize('_max_shifts', (1, 1_000, None))
    def test_accuracy(self, _max_shifts):

        assert _cond_max_shifts(_max_shifts) == (_max_shifts or 1_000)






