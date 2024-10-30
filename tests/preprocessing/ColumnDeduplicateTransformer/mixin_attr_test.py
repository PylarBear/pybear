# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

import pytest



# this test module shows which mixins have which attrs/methods (if any).
# if the mixins dont have these attrs/methods, then they are built in
# the module.
#     n_features_in_
#     feature_names_in_
#     n_samples_seen_
#     get_feature_names_out()
#     _validate_data
#     set_outputs

# use a dataframe to make sure the feature name attrs get exposed!

# 24_10_02_12_40_00
# TransformerMixin alone does not expose any of these, it only exposes
# fit_transform.
# BaseEstimator alone exposes _validate_data, get_params, set_params.
# _validate_data exposes n_features_in_ and feature_names_in_.
# OneToOneFeatureMixin alone exposes get_feature_names_out().

# So what happens with OneHotEncoder, which changes the columns?
# n_features_in_ AND feature_names_in_ expose the values before transform.
# To get feature names after transform, use get_feature_names_out(),
# which is a one-off that overwrites get_feature_names_out() from
# BaseEstimator.






class Fixtures:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 5)


    @staticmethod
    @pytest.fixture(scope='module')
    def X_pd(_X_factory, _master_columns, _shape):

        return _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='pd',
            _dtype='flt',
            _columns=_master_columns.copy()[:_shape[1]]
        )


    # create dummy classes that take the mixin and have a fit method
    # BaseEstimator & TransformerMixin do not have fit method.

    @staticmethod
    @pytest.fixture()
    def MockWithBaseEstimator():

        class MockBaseEstimator(BaseEstimator):

            def __init__(cls):
                pass

            def _reset(cls):
                if hasattr(cls, 'is_fitted'):
                    del cls.is_fitted

            def fit(cls, X, y=None):
                cls._reset()
                # this exposes n_features_in_ and feature_names_in_
                X = cls._validate_data(
                    X,
                    accept_sparse=("csr", "csc", "coo"),
                    dtype=None,
                    force_all_finite="allow-nan",
                    reset=True,
                )
                cls.is_fitted = True
                return cls

        return MockBaseEstimator


    @staticmethod
    @pytest.fixture()
    def MockWithTransformerMixin():

        class MockTransformerMixin(TransformerMixin):

            def __init__(cls):
                pass

            def _reset(cls):
                if hasattr(cls, 'is_fitted'):
                    del cls.is_fitted

            def fit(cls, X, y=None):
                cls._reset()
                cls.is_fitted = True
                return cls

        return MockTransformerMixin


    @staticmethod
    @pytest.fixture()
    def MockWithOneToOneFeatureMixin():

        class MockOneToOneFeatureMixin(OneToOneFeatureMixin):

            def __init__(cls):
                pass

            def _reset(cls):
                if hasattr(cls, 'is_fitted'):
                    del cls.is_fitted

            def fit(cls, X, y=None):
                cls._reset()
                cls.is_fitted = True
                return cls

        return MockOneToOneFeatureMixin


class TestBaseEstimator(Fixtures):

    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_before_fit(self, attr, MockWithBaseEstimator):

        if attr in ['_validate_data', 'set_params', 'get_params']:
            assert hasattr(MockWithBaseEstimator, attr)
        else:
            assert not hasattr(MockWithBaseEstimator, attr)


    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_after_fit(self, attr, MockWithBaseEstimator, X_pd):

        Fitted = MockWithBaseEstimator().fit(X_pd)

        if attr in [
            '_validate_data', 'n_features_in_', 'feature_names_in_',
            'set_params', 'get_params'
        ]:
            assert hasattr(Fitted, attr)
        else:
            assert not hasattr(Fitted, attr)


class TestTransformerMixin(Fixtures):

    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_before_fit(self, attr, MockWithTransformerMixin):

        assert not hasattr(MockWithTransformerMixin, attr)


    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_after_fit(self, attr, MockWithTransformerMixin, X_pd):

        Fitted = MockWithTransformerMixin().fit(X_pd)

        assert not hasattr(Fitted, attr)





class TestOneToOneFeatureMixin(Fixtures):

    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_before_fit(self, attr, MockWithOneToOneFeatureMixin):

        if attr == 'get_feature_names_out':
            assert hasattr(MockWithOneToOneFeatureMixin, attr)
        else:
            assert not hasattr(MockWithOneToOneFeatureMixin, attr)


    @pytest.mark.parametrize('attr',
        ('n_features_in_', 'feature_names_in_', 'n_samples_seen_',
         'get_feature_names_out', '_validate_data', 'set_params', 'get_params')
    )
    def test_after_fit(self, attr, MockWithOneToOneFeatureMixin, X_pd):

        Fitted = MockWithOneToOneFeatureMixin().fit(X_pd)

        if attr == 'get_feature_names_out':
            assert hasattr(Fitted, attr)
        else:
            assert not hasattr(Fitted, attr)







