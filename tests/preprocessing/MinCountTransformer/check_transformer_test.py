# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


from pybear.preprocessing import MinCountTransformer
from sklearn.utils.estimator_checks import (
    check_transformer_general,
    check_transformers_unfitted,
    check_transformer_data_not_an_array,
    check_transformer_preserve_dtypes,
    check_transformer_n_iter,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas
)





class TestSKLearnCheckTransformer:


    def test_transformers_unfitted(self):
        # this tests if Exception raised when transform() without fit()
        check_transformers_unfitted(
            'MinCountTransformer',
            MinCountTransformer(count_threshold=5)
        )


    def test_transformer_general(self):

        # err_msg = 'fit_transform and transform outcomes not consistent
        # in MinCountTransformer(count_threshold=5)'

        # violates sklearn API on example axis
        with pytest.raises(ValueError):
            check_transformer_general(
                'MinCountTransformer',
                MinCountTransformer(count_threshold=5)
            )


    def test_transformer_data_not_an_array(self):

        # this fails because it passes an object that MCT cannot convert to ndarray
        with pytest.raises(TypeError):
            check_transformer_data_not_an_array(
                'MinCountTransformer',
                MinCountTransformer(count_threshold=5)
            )


    def test_transformer_preserve_dtypes(self):
        check_transformer_preserve_dtypes(
            'MinCountTransformer',
            MinCountTransformer(count_threshold=5)
        )


    def test_check_transformer_n_iter(self):
        check_transformer_n_iter(
            'MinCountTransformer',
            MinCountTransformer(count_threshold=5)
    )


    def test_check_transformer_get_feature_names_out(self):
        # looks for certain verbage in error if len(input_features) does not
        # match n_features_in_, and if output dtype is object
        check_transformer_get_feature_names_out(
            'MinCountTransformer',
            MinCountTransformer(count_threshold=5)
        )


    def test_check_transformer_get_feature_names_out_pandas(self):
        # looks for certain verbage in error if 'input_features' does not
        # match feature_names_in_ if MCT was fit on a dataframe
        check_transformer_get_feature_names_out_pandas(
            'MinCountTransformer',
            MinCountTransformer(count_threshold=5)
        )






