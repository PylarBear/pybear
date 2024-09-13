# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from typing_extensions import Union
import numpy as np
import dask.array as da


@pytest.fixture(scope='module')
def mock_estimator_test_fixture():


    class MockEstimator:

        param_a: Union[int, float]
        param_b: Union[int, float]

        def __init__(
                self,
                param_a=5,
                param_b=np.pi
            ):

            self.param_a = param_a
            self.param_b = param_b


        def get_params(self, deep=True):
            return {
                'param_a': self.param_a,
                'param_b': self.param_b
            }


        def set_params(self, **params):

            if 'param_a' in params:
                self.param_a = params['param_a']

            if 'param_b' in params:
                self.param_b = params['param_b']

            return self


        def partial_fit(self, X):

            if isinstance(X, da.core.Array):

                _min_dim = min(X.shape)
                X += self.param_a
                _square_matrix = X[:_min_dim, :_min_dim]

            else:
                X = np.array(X)
                _min_dim = min(X.shape)
                X += self.param_a
                _square_matrix = X[:_min_dim, :_min_dim]

            _square_matrix -= self.param_b

            if hasattr(self, '_square_matrix'):
                self._square_matrix += _square_matrix
            else:
                self._square_matrix = _square_matrix


        def fit(self, X, y):

            if isinstance(X, da.core.Array):

                _min_dim = min(X.shape)
                X += self.param_a
                _square_matrix = X[:_min_dim, :_min_dim]

            else:
                X = np.array(X)
                _min_dim = min(X.shape)
                X += self.param_a
                _square_matrix = X[:_min_dim, :_min_dim]

            _square_matrix -= self.param_b

            self._square_matrix = _square_matrix


        def predict_proba(self, X):

            _rows = X.shape[0]

            if isinstance(X, da.core.Array):
                return da.random.uniform(0, 1, (_rows, 2))
            else:
                return np.random.uniform(0, 1, (_rows, 2))


        def predict(self, X):
            return (self.predict_proba(X)[:-1] >= 0.5).astype(np.uint8)


        def score(self, X, y):   # needs two args to satisify sklearn
            return self._square_matrix[0].sum() / self._square_matrix.sum()


    return MockEstimator()































