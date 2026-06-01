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

from pybear.preprocessing import NanStandardizer as NS



class TestSKLearnCheckTransformer:


    sk_version = sklearn.__version__


    def test_transformers_unfitted(self):
        # this tests if Exception raised when transform() without fit()

        # AssertionError: The unfitted transformer NanStandardizer does
        # not raise an error when transform is called. Perhaps use
        # check_is_fitted in transform.
        # Fails this test because NS is always fitted.
        with pytest.raises(AssertionError):
            check_transformers_unfitted(
                'NanStandardizer',
                NS()
            )


    def test_transformer_general(self):

        # AssertionError: The transformer NanStandardizer does not raise
        # an error when the number of features in transform is different
        # from the number of features in fit.
        # Fails this test because NS does not use base.validate_data,
        # which mean it does not get & validate n_features_in_
        with pytest.raises(AssertionError):
            check_transformer_general(
                'NanStandardizer',
                NS()
            )


    def test_transformer_preserve_dtypes(self):
        check_transformer_preserve_dtypes(
            'NanStandardizer',
            NS()
        )


    def test_check_transformer_get_feature_names_out(self):
        # looks for certain verbiage in error if len(input_features) does not
        # match n_features_in_, and if output dtype is object

        if float(self.sk_version[0:3]) >= 1.6:
            # AttributeError: 'NanStandardizer' object has no attribute
            # 'get_feature_names_out'
            # Fails this test because NS does not use base.validate_data,
            # which mean it does not get features_names_in_
            err_msg = f"'NanStandardizer' object has no attribute 'get_feature_names_out'"
            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out(
                    'NanStandardizer',
                    NS()
                )
        else:
            err_msg = f"'NanStandardizer' object has no attribute '_get_tags'"

            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out(
                    'NanStandardizer',
                    NS()
                )


    def test_check_transformer_get_feature_names_out_pandas(self):
        # looks for certain verbiage in error if 'input_features' does not
        # match feature_names_in_ if NS was fit on a dataframe

        if float(self.sk_version[0:3]) >= 1.6:
            # AttributeError: 'NanStandardizer' object has no attribute
            # 'get_feature_names_out'
            # Fails this test because NS does not use base.validate_data,
            # which mean it does not get features_names_in_
            err_msg = f"'NanStandardizer' object has no attribute 'get_feature_names_out'"
            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out_pandas(
                    'NanStandardizer',
                    NS()
                )
        else:
            err_msg = f"'NanStandardizer' object has no attribute '_get_tags'"

            with pytest.raises(AttributeError, match=re.escape(err_msg)):
                check_transformer_get_feature_names_out_pandas(
                    'NanStandardizer',
                    NS()
                )



