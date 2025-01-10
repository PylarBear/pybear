# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.base.mixins._SetParamsMixin import SetParamsMixin

import numpy as np

import pytest




pytest.skip(reason='pizza not started, not finished', allow_module_level=True)





class Fixtures:


    @staticmethod
    @pytest.fixture(scope='function')
    def _shape():
        return (10, 5)


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyEstimator():

        class Foo(SetParamsMixin):

            def __init__(self):
                self._is_fitted = False   # <===== leading under
                self.dum_attr_ = 1      # <===== trailing under
                self.bananas = 7
                self.fries = 9
                self.ethanol = 5
                self.apples = 4


            def reset(self):

                self._is_fitted = False


            def fit(self, X, y=None):
                self.reset()
                self._is_fitted = True
                return self


            def score(self, X, y=None):
                return np.random.uniform(0, 1)


            def predict(self, X):

                assert self._is_fitted

                return np.random.randint(0, 2, X.shape[0])


        return Foo  # <====== not initialized



    @staticmethod
    @pytest.fixture(scope='function')
    def DummyTransformer():

        class Bar(SetParamsMixin):

            def __init__(self):
                self._is_fitted = False   #  <==== leading under
                self.this_attr_ = 1    # <====== tralling under
                self.tbone = 3
                self.wings = 7
                self.bacon = 9
                self.sausage = 5
                self.hambone = 4


            def reset(self):
                try:
                    delattr(self, '_fill_value')
                except:
                    pass
                self._is_fitted = False


            def fit(self, X, y=None):
                self.reset()
                self._fill_value = np.random.uniform(0, 1)
                self._is_fitted = True
                return self


            def transform(self, X):

                assert self._is_fitted

                return np.full(X.shape, self._fill_value)


            def fit_transform(self, X, y=None):
                return self.fit(X, y=y).transform(X)


        return Bar  # <====== not initialized



    @staticmethod
    @pytest.fixture(scope='function')
    def DummyGridSearch(DummyEstimator):

        class Baz(SetParamsMixin):

            def __init__(
                self,
                estimator,
                param_grid,
                *,
                scoring='balanced_accuracy',
                refit=False
            ):
                self.estimator = estimator
                self.param_grid = param_grid
                self.scoring = scoring
                self.refit = refit
                self.apricots = False


            def fit(self, X):

                self.best_params_ = {}

                for _param in self.param_grid:
                    self.best_params_[_param] = \
                        np.random.choice(self.param_grid[_param])


        return Baz





class TestSetParamsMixin:

    pass











