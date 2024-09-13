# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing.MinCountTransformer._shared._validation. \
    _val_n_features_in import _val_n_features_in




class TestValNFeaturesIn:

    @pytest.mark.parametrize('_n_features_in',
        (True, None, min, [1], (1,), {1,2}, {'a':1}, lambda x: x, 'junk', 3.14)
    )
    def test_typeerror_junk_n_features_in(self, _n_features_in):
        with pytest.raises(TypeError):
            _val_n_features_in(_n_features_in)


    @pytest.mark.parametrize('_n_features_in',
        (-1, 0)
    )
    def test_valueerror_n_features_In(self, _n_features_in):
        with pytest.raises(ValueError):
            _val_n_features_in(_n_features_in)


    @pytest.mark.parametrize('_n_features_in', (1, 2, 3)
    )
    def test_accepts_good_count_threshold(self, _n_features_in):
        _val_n_features_in(_n_features_in)











