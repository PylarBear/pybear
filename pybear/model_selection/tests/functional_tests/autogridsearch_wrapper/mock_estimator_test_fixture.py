# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union
import numpy as np




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


    def score(self, one, two):   # needs two args to satisify sklearn
        return np.linalg.det(self._square_matrix)


    def fit(self, X, y):

        X = np.array(X)

        _square_matrix = np.matmul(
            X + self.param_a, X.transpose()
        )
        _square_matrix *= (y - self.param_b)

        self._square_matrix = _square_matrix







































