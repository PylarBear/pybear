# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextPadder._validation._n_features import \
    _val_n_features



class TestValNFeatures:


    @pytest.mark.parametrize(f'junk_n_features',
        (-2.7, 2.7, True, False, None, 'trash', [0,1], {1,2}, (1,), {"A":1},
         lambda x: x)
    )
    def test_rejects_non_int(self, junk_n_features):

        with pytest.raises(TypeError):
            _val_n_features(junk_n_features)


    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            _val_n_features(-1)


    @pytest.mark.parametrize('n_features', (0, 1, 2, 3))
    def test_accepts_good(self, n_features):

        _val_n_features(n_features)



