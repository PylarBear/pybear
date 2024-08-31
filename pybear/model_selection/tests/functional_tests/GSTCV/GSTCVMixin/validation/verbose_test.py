# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


from model_selection.GSTCV._GSTCVMixin._validation._verbose \
    import _validate_verbose


class TestValidateVerbose:


    @pytest.mark.parametrize('junk_verbose',
        (
        None, 'trash', min, int, [0,1], (1,0), {1,0}, {'a': 1}, lambda x: x,
        float('inf'), float('-inf')
        )
    )
    def test_rejects_non_numeric(self, junk_verbose):

        with pytest.raises(TypeError):
            _validate_verbose(junk_verbose)


    @pytest.mark.parametrize('bad_verbose',
        (-4, -1)
    )
    def test_rejects_negative(self, bad_verbose):
        with pytest.raises(ValueError):
            _validate_verbose(bad_verbose)


    def test_bools(self):
        assert _validate_verbose(True) == 10
        assert _validate_verbose(False) == 0


    def test_floats(self):
        assert _validate_verbose(0.124334) == 0
        assert _validate_verbose(3.14) == 3
        assert _validate_verbose(8.8888) == 9


    @pytest.mark.parametrize('good_int', (0, 1, 5, 200))
    def test_ints(self, good_int):
        assert _validate_verbose(good_int) == min(good_int, 10)




