# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.base.mixins._ReprMixin import ReprMixin
from pybear.base.mixins._GetParamsMixin import GetParamsMixin

import numpy as np

import pytest



class TestReprMixin:


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyTransformer():

        class DummyTransformer(ReprMixin, GetParamsMixin):

            # FeatureMixin should provide
            # get_feature_names_out(input_features)
            # _check_n_features(X, self.n_features_in, reset)
            # _check_feature_names_in(X, reset)

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

                self._is_fitted = True

                return self


            def fit(self, X, y=None):
                self.reset()
                return self.partial_fit(X, y)


            def transform(self, X):

                assert self._is_fitted

                return np.full(X.shape, np.e)


        return DummyTransformer  # <====== not initialized



    def test_repr_mixin(self, DummyTransformer):

        cls = DummyTransformer()

        X = np.random.randint(0, 10, (20, 13))

        print(cls.fit(X))




















