# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from sklearn.utils.estimator_checks import (
    check_transformers_unfitted,
    check_transformer_general,
    check_transformer_preserve_dtypes,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas
)


pytest.skip(reason=f"not finished", allow_module_level=True)


class TestSKLearnCheckTransformer:


    def test_transformers_unfitted(self):
        # this tests if Exception raised when transform() without fit()
        check_transformers_unfitted(
            'SlimPoly',
            SlimPoly()
        )


    def test_transformer_general(self):

        check_transformer_general(
            'SlimPoly',
            SlimPoly()
        )


    def test_transformer_preserve_dtypes(self):
        check_transformer_preserve_dtypes(
            'SlimPoly',
            SlimPoly()
        )


    def test_check_transformer_get_feature_names_out(self):
        # looks for certain verbage in error if len(input_features) does not
        # match n_features_in_, and if output dtype is object
        check_transformer_get_feature_names_out(
            'SlimPoly',
            SlimPoly()
        )


    def test_check_transformer_get_feature_names_out_pandas(self):
        # looks for certain verbage in error if 'input_features' does not
        # match feature_names_in_ if MCT was fit on a dataframe
        check_transformer_get_feature_names_out_pandas(
            'SlimPoly',
            SlimPoly()
        )





