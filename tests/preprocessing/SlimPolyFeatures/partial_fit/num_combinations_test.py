# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from pybear.preprocessing.SlimPolyFeatures._partial_fit._num_combinations import \
    _val_num_combinations



import pytest



class TestValNumCombinations:

    # def _val_num_combinations(
    #     n_features_in_: int,
    #     _n_poly_features: int,
    #     _min_degree: int,
    #     _max_degree: int,
    #     _intx_only: bool
    # ) -> None:


    @pytest.mark.parametrize('_n_poly_features', (2**64, ))
    def test_rejects_too_many_features_out(self, _n_poly_features):

        with pytest.raises(ValueError):
            _val_num_combinations(
                n_features_in_=100,
                _n_poly_features=_n_poly_features,
                _min_degree=1,
                _max_degree=100,
                _intx_only=False
            )


    @pytest.mark.parametrize('_n_poly_features', (1, 2**4, 2**8, 2**16, 2**32))
    def test_accepts_otherwise(self, _n_poly_features):
        _val_num_combinations(
            n_features_in_=100,
            _n_poly_features=_n_poly_features,
            _min_degree=1,
            _max_degree=100,
            _intx_only=False
        )







