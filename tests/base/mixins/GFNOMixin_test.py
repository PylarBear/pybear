# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.base import GFNOMixin

import uuid
import numpy as np

import pytest



# pizza this is going to be folded into FeatureMixin


class TestGFNOMixin:


    # the workhorse of the mixin is pybear.base.get_feature_names_out().
    # that module is tested in get_feature_names_out_test.
    # only test that the mixin works here.


    @staticmethod
    @pytest.fixture(scope='function')
    def _shape():
        return (1_000, 10)


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyTransformer():


        def foo(
            feature_names_in_,
            n_features_in_
        ):

            class DummyTransformer(GFNOMixin):

                def __init__(self):
                    self.feature_names_in_ = feature_names_in_
                    self.n_features_in_ = n_features_in_

                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    return X


            return DummyTransformer()  # <====== initialize here!


        return foo



    @pytest.mark.parametrize('_input_features_is_passed', (True, False))
    @pytest.mark.parametrize('_feature_names_in_is_passed', (True, False))
    @pytest.mark.parametrize('_n_features_in_is_passed', (True, False))
    def test_gfno_mixin(
        self, _shape, DummyTransformer, _input_features_is_passed,
        _feature_names_in_is_passed, _n_features_in_is_passed
    ):

        dum_header = np.array(
            [str(uuid.uuid4())[:4] for _ in range(_shape[1])],
            dtype=object
        )


        if _input_features_is_passed:
            _input_features = dum_header.copy()
        else:
            _input_features = None

        if _feature_names_in_is_passed:
            feature_names_in_ = dum_header.copy()
        else:
            feature_names_in_ = None

        if _n_features_in_is_passed:
            n_features_in_ = _shape[1]
        else:
            n_features_in_ = None


        TestClass = DummyTransformer(feature_names_in_, n_features_in_)


        if _n_features_in_is_passed:
            out = TestClass.get_feature_names_out(input_features=_input_features)
        elif not _n_features_in_is_passed:
            with pytest.raises(AssertionError):
                TestClass.get_feature_names_out(input_features=_input_features)
            pytest.skip(reason=f"cant do more tests after exception")


        assert out.dtype == object

        if _feature_names_in_is_passed:
            assert np.array_equal(out, dum_header)
        elif not _feature_names_in_is_passed:
            if _input_features_is_passed:
                assert np.array_equal(out, dum_header)
            elif not _input_features_is_passed:
                boilerplate = [f'x{i}' for i in range(_shape[1])]
                ref = np.array(boilerplate, dtype=object)
                assert np.array_equal(out, ref)

















