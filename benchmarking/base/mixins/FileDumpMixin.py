# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from pybear.base.mixins._FileDumpMixin import FileDumpMixin
from pybear.base.mixins._FitTransformMixin import FitTransformMixin



# this file is a sandbox for testing FileDumpMixin


class Cls(FitTransformMixin, FileDumpMixin):


    def __init__(self, fill:bool=True):

        self._is_fitted = False
        self.fill = fill


    def __pybear_is_fitted__(self):
        return getattr(self, '_is_fitted', None) is True


    def reset(self):
        self._is_fitted = False

        return self


    def _validate(self, OBJ, name):
        if not isinstance(
            OBJ, (Sequence, set, np.ndarray, pd.DataFrame, pl.DataFrame)
        ) and not hasattr(OBJ, 'toarray'):

            raise TypeError(
                f"'{name}' must be a python built-in, numpy array, pandas "
                f"series/dataframe, or polars series/dataframe."
            )


    def fit(self, X, y=None):

        self._validate(X, 'X')
        if y is not None:
            self._validate(y, 'y')

        self.n_features_in_ = X.shape[1]

        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        self.fill_value = np.random.randint(1, 10)

        self._is_fitted = True

        return self


    def transform(self, X):

        return np.full(X.shape if hasattr(X, 'shape') else (5,5), self.fill)



if __name__ == '__main__':

    _shape = (5, 10)

    X = np.random.choice(list('abcedefghijkl'), _shape)

    trfm = Cls().fit(X)

    trfm.dump_to_txt(X)







