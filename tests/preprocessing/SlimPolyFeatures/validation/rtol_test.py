# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.SlimPolyFeatures._validation._rtol \
    import _val_rtol


import pytest



class TestRtol:

    @pytest.mark.parametrize('junk_rtol',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, min, lambda x: x)
    )
    def test_rejects_junk(self, junk_rtol):
        # this is handled by np.allclose, let it raise whatever
        with pytest.raises(Exception):
            _val_rtol(junk_rtol)


    @pytest.mark.parametrize('bad_rtol',
        (-1, True, False)
    )
    def test_accepts_good(self, bad_rtol):
        with pytest.raises(TypeError):
            _val_rtol(bad_rtol)


    @pytest.mark.parametrize('good_rtol',
        (0, 1e-6, 0.1, 1, 3.14)
    )
    def test_accepts_good(self, good_rtol):
        _val_rtol(good_rtol)






