# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import sklearn

from sklearn.utils.estimator_checks import (
    check_transformers_unfitted,
    check_transformer_general,
    check_transformer_preserve_dtypes,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas
)

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT



class TestSKLearnCheckTransformer:


    sk_version = sklearn.__version__


    def test_transformers_unfitted(self):
        # this tests if Exception raised when transform() without fit()
        check_transformers_unfitted(
            'ColumnDeduplicateTransformer',
            CDT()
        )


    def test_transformer_general(self):

        check_transformer_general(
            'ColumnDeduplicateTransformer',
            CDT()
        )


    def test_transformer_preserve_dtypes(self):
        check_transformer_preserve_dtypes(
            'ColumnDeduplicateTransformer',
            CDT()
        )


    def test_check_transformer_get_feature_names_out(self):
        # looks for certain verbiage in error if len(input_features) does not
        # match n_features_in_, and if output dtype is object

        if float(self.sk_version[0:3]) >= 1.6:
            check_transformer_get_feature_names_out(
                'ColumnDeduplicateTransformer',
                CDT()
            )
        else:
            err_msg = (f"'ColumnDeduplicateTransformer' object has no "
                       f"attribute '_get_tags'")

            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out(
                    'ColumnDeduplicateTransformer',
                    CDT()
                )


    def test_check_transformer_get_feature_names_out_pandas(self):
        # looks for certain verbiage in error if 'input_features' does not
        # match feature_names_in_ if CDT was fit on a dataframe

        if float(self.sk_version[0:3]) >= 1.6:
            check_transformer_get_feature_names_out_pandas(
                'ColumnDeduplicateTransformer',
                CDT()
            )
        else:
            err_msg = (f"'ColumnDeduplicateTransformer' object has no "
                       f"attribute '_get_tags'")

            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out_pandas(
                    'ColumnDeduplicateTransformer',
                    CDT()
                )



