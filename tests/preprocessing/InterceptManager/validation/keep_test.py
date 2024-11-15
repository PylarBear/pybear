# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._validation._keep import _val_keep


import pytest



class TestKeep:


    @pytest.mark.parametrize('junk_keep',
        (-1, 0, 1, 3.14, True, False, None, [0,1], {0,1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_keep):
        with pytest.raises(ValueError):
            _val_keep(junk_keep)


    @pytest.mark.parametrize('bad_keep',
        ('trash', 'garbage', 'junk', {0:1}, {0:'junk'})
    )
    def test_rejects_bad(self, bad_keep):
        with pytest.raises(ValueError):
            _val_keep(bad_keep)


    @pytest.mark.parametrize('good_keep',
        ('first', 'last', 'random', 'none', {'Intercept': 1})
    )
    def test_accepts_good(self, good_keep):
        _val_keep(good_keep)










