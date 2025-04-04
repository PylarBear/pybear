# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing._SlimPolyFeatures._validation._atol \
    import _val_atol


import pytest



class TestAtol:

    @pytest.mark.parametrize('junk_atol',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, min, lambda x: x)
    )
    def test_rejects_junk(self, junk_atol):
        # this is handled by np.allclose, let it raise whatever
        with pytest.raises(Exception):
            _val_atol(junk_atol)


    @pytest.mark.parametrize('bad_atol',
        (-1, True, False)
    )
    def test_rejects_bad(self, bad_atol):
        with pytest.raises(ValueError):
            _val_atol(bad_atol)


    @pytest.mark.parametrize('good_atol',
        (0, 1e-6, 0.1, 1, 3.14)
    )
    def test_accepts_good(self, good_atol):
        _val_atol(good_atol)






