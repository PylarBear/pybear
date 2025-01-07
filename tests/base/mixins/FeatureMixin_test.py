# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.base.mixins._FeatureMixin import FeatureMixin
from pybear.base.exceptions import NotFittedError

import uuid
import numpy as np
import pandas as pd

import pytest




class TestFeatureMixin:


    # the workhorses of this mixin are
    # pybear.base.check_n_features()
    # pybear.base.check_feature_names()
    # pybear.base.get_feature_names_out().
    # those modules are tested in
    # check_n_features_test
    # check_feature_names_test
    # get_feature_names_out_test.
    # only test that the mixin works here.


    @staticmethod
    @pytest.fixture(scope='function')
    def _shape():
        return (
            int(np.random.randint(2, 1_000)),
            int(np.random.randint(2, 20))
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='function')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_pd(_X_np, _columns):
        return pd.DataFrame(data=_X_np, columns=_columns)


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyTransformer():

        class DummyTransformer(FeatureMixin):

            # FeatureMixin should provide
            # get_feature_names_out(input_features)
            # _check_n_features(X, self.n_features_in, reset)
            # _check_feature_names(X, reset)

            # feature axis of X is not altered during transform

            def __init__(self):
                self._is_fitted = False

            def reset(self):
                try:
                    delattr(self, '_random_fill')
                except:
                    pass

                self._is_fitted = False

            def partial_fit(self, X, y=None):

                self._check_n_features(
                    X,
                    reset=not hasattr(self, '_random_fill')
                )

                self._check_feature_names(
                    X,
                    reset=not hasattr(self, '_random_fill')
                )

                self._random_fill = np.random.uniform(0, 1)

                self._is_fitted = True

                return self


            def fit(self, X, y=None):
                self.reset()
                return self.partial_fit(X, y)


            def transform(self, X):

                assert self._is_fitted

                # X must have same n_features as at fit for get_feature_names_out()
                assert X.shape[1] == self.n_features_in

                return np.full(X.shape, self._random_fill)


        return DummyTransformer  # <====== not initialized


    @pytest.mark.parametrize('_input_features_is_passed',
        ('true_valid', 'true_invalid', 'false')
    )
    @pytest.mark.parametrize('_feature_names_are_passed', (True, False))
    def test_gfno(
        self, _shape, _columns, _X_np, _X_pd, DummyTransformer,
        _input_features_is_passed, _feature_names_are_passed
    ):

        if _input_features_is_passed == 'true_valid':
            _input_features = _columns.copy()
        elif _input_features_is_passed == 'true_invalid':
            _input_features = [str(uuid.uuid4)[:5] for _ in range(_shape[1])]
        elif _input_features_is_passed == 'false':
            _input_features = None
        else:
            raise Exception

        if _feature_names_are_passed:
            _X_wip = _X_pd
        else:
            _X_wip = pd.DataFrame(
                data=_X_np,
                columns=None
            )


        TestClass = DummyTransformer()

        # excepts when not fitted yet
        with pytest.raises(NotFittedError):
            TestClass.get_feature_names_out(input_features=_input_features)

        TestClass.fit(_X_wip)

        # the only thing that should raise an exception after fit is if invalid
        # feature names are passed to input_features when first fit saw feature
        # names. so get that raise out of the way then handle all the other tests.
        if _feature_names_are_passed and _input_features_is_passed == 'true_invalid':
            with pytest.raises(ValueError):
                TestClass.get_feature_names_out(input_features=_input_features)
            pytest.skip(reason=f"cant do anymore tests after exception")
        else:
            out = TestClass.get_feature_names_out(input_features=_input_features)

        assert isinstance(out, np.ndarray)
        assert out.dtype == object

        if _feature_names_are_passed:
            if _input_features_is_passed == 'true_valid':
                assert np.array_equal(out, _columns)
            elif _input_features_is_passed == 'false':
                assert np.array_equal(out, _columns)
        elif not _feature_names_are_passed:
            if _input_features_is_passed == 'true_valid':
                assert np.array_equal(out, _columns)
            elif _input_features_is_passed == 'true_invalid':
                assert np.array_equal(out, _input_features)
            elif _input_features_is_passed == 'false':
                boilerplate = [f'x{i}' for i in range(_shape[1])]
                ref = np.array(boilerplate, dtype=object)
                assert np.array_equal(out, ref)


    @pytest.mark.parametrize('X_format', ('np', 'pd'))
    def test_n_features_in_(
        self, DummyTransformer, X_format, _X_np, _X_pd, _shape
    ):

        TestClass = DummyTransformer()

        # excepts when not fitted yet
        with pytest.raises(AttributeError):
            getattr(TestClass, 'n_features_in_')

        if X_format == 'np':
            TestClass.fit(_X_np)
        elif X_format == 'pd':
            TestClass.fit(_X_pd)
        else:
            raise Exception

        # n_features_in_ should always be exposed no matter what valid container
        assert getattr(TestClass, 'n_features_in_') == _shape[1]


    @pytest.mark.parametrize('X_format', ('np', 'pd_w_header', 'pd_no_header'))
    def test_features_names_in_(
        self, DummyTransformer, X_format, _X_np, _X_pd, _columns
    ):

        TestClass = DummyTransformer()

        # excepts when not fitted yet
        with pytest.raises(AttributeError):
            getattr(TestClass, 'feature_names_in_')

        if X_format == 'np':
            TestClass.fit(_X_np)
        elif X_format == 'pd_w_header':
            TestClass.fit(_X_pd)
        elif X_format == 'pd_no_header':
            __ = pd.DataFrame(data=_X_np)
            TestClass.fit(__)
        else:
            raise Exception

        # feature_names_in_ should only be exposed for container with valid header
        if X_format == 'pd_w_header':
            out = getattr(TestClass, 'feature_names_in_')
            assert isinstance(out, np.ndarray)
            assert out.dtype == object
            assert np.array_equal(out, _columns)
        else:
            with pytest.raises(AttributeError):
                getattr(TestClass, 'feature_names_in_')
















