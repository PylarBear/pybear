# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.model_selection.GSTCV._GSTCVMixin._param_conditioning._verbose \
    import _cond_verbose



class TestCondVerbose:


    def test_bools(self):
        assert _cond_verbose(True) == 10
        assert _cond_verbose(False) == 0


    def test_floats(self):
        assert _cond_verbose(0.124334) == 0
        assert _cond_verbose(3.14) == 3
        assert _cond_verbose(8.8888) == 9


    @pytest.mark.parametrize('good_int', (0, 1, 5, 200))
    def test_ints(self, good_int):
        assert _cond_verbose(good_int) == min(good_int, 10)





