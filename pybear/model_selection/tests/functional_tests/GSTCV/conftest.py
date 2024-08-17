# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import time
import numpy as np
import pandas as pd
from dask import compute, dataframe as ddf
import dask_expr._collection as ddf2

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict
from sklearn.model_selection import ParameterGrid

from xgboost import XGBClassifier




@pytest.fixture(scope='module')
def _cv_results_template(request):

    _n_splits = request.param['_n_splits']
    _n_rows = request.param['_n_rows']
    _scorer_names = request.param['_scorer_names']
    _grids = request.param['_grids']
    _return_train_score = request.param['_return_train_score']
    _fill_param_columns = request.param['_fill_param_columns']

    # build _scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    try:
        iter(_scorer_names)
        if isinstance(_scorer_names, (str, dict)):
            raise Exception
    except:
        raise Exception(
            f"'LIST_OF_SCORERS' must be an iterable of scorer names")

    _scorer = {}
    for _name in _scorer_names:
        if _name not in master_scorer_dict:
            raise ValueError(f"'{_name}' is not an allowed scorer")

        _scorer[_name] = master_scorer_dict[_name]

    # END build _scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    col_template = lambda _dtype: np.ma.masked_array(
        np.empty(_n_rows),
        mask=True,
        dtype=_dtype
    )

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    a = {
        'mean_fit_time': col_template(np.float64),
        'std_fit_time': col_template(np.float64),
        'mean_score_time': col_template(np.float64),
        'std_score_time': col_template(np.float64)
    }
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    b = {}
    for _grid in _grids:
        for _param in _grid:
            # this will overwrite any identical, preventing duplicate
            if isinstance(_grid[_param][0], bool):
                b[f'param_{_param}'] = col_template(bool)
            else:
                try:
                    float(_grid[_param][0])
                    b[f'param_{_param}'] = col_template(float)
                except:
                    b[f'param_{_param}'] = col_template(str)

        b = b | {'params': col_template(object)}

    if _fill_param_columns:
        row_idx = 0
        for _grid in _grids:
            for _permutation in ParameterGrid(_grid):
                # ParameterGrid lays out permutations in the same order as pybear.permuter

                b['params'][row_idx] = _permutation

                for _param in _grid:
                    if f'param_{_param}' not in b:
                        raise Exception
                    b[f'param_{_param}'][row_idx] = _permutation[_param]

                row_idx += 1


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    c = {}
    for metric in _scorer:
        suffix = 'score' if len(_scorer) == 1 else f'{metric}'

        if len(_scorer) == 1:
            c[f'best_threshold'] = col_template(np.float64)
        else:
            c[f'best_threshold_{metric}'] = col_template(np.float64)

        for split in range(_n_splits):
            c[f'split{split}_test_{suffix}'] = col_template(np.float64)

        c[f'mean_test_{suffix}'] = col_template(np.float64)
        c[f'std_test_{suffix}'] = col_template(np.float64)
        c[f'rank_test_{suffix}'] = col_template(np.uint32)

        if _return_train_score is True:

            for split in range(_n_splits):
                c[f'split{split}_train_{suffix}'] = col_template(np.float64)

            c[f'mean_train_{suffix}'] = col_template(np.float64)
            c[f'std_train_{suffix}'] = col_template(np.float64)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    return a | b | c




@pytest.fixture(scope='module')
def _mock_classifier():

    return XGBClassifier
    #     (
    #     tree_method='hist',
    #     max_depth=3,
    #     learning_rate=0.1,
    #     n_estimators=100
    # )

    # class MockCLF:
    #
    #     def __init__(self, ctr, sleep=0, C=1):
    #         self.ctr = ctr
    #         self._is_fitted = 0
    #         self.sleep = sleep
    #         self.C = C
    #
    #     @property
    #     def ctr_(self):
    #         return self.ctr
    #
    #     def get_params(self, deep=True):
    #         return {'ctr': self.ctr, 'sleep': self.sleep, 'C': self.C}
    #
    #
    #     def set_params(self, **params):
    #         for param in params:
    #             if param not in ['ctr', 'sleep', 'C']:
    #                 raise ValueError(f"unknown param '{param}'")
    #             setattr(self, param, params[param])
    #         return self
    #
    #
    #     def is_fitted(self):
    #         return self._is_fitted > 0
    #
    #
    #     def fit(self, X, y=None, **fit_kwargs):
    #
    #         self.n_features_in_ = compute(*X.shape)[0]
    #
    #         self.feature_names_in_ = None
    #         if isinstance(X, (pd.core.frame.DataFrame, ddf.core.DataFrame, ddf2.DataFrame)):
    #             self.feature_names_in_ = np.array(X.columns)
    #
    #         self.classes_ = np.unique(y)
    #
    #         time.sleep(self.sleep)
    #         self._is_fitted = 1
    #         self.ctr = 1
    #         return self
    #
    #
    #     def partial_fit(self, X, y=None, **fit_kwargs):
    #
    #         time.sleep(self.sleep)
    #         self._is_fitted += 1
    #         self.ctr += 1
    #         return self
    #
    #
    #     def score(self, *args, **kwargs):
    #         return np.exp(-10000*((self.C-0.001)**2))
    #
    #
    #     def transform(self, X, y=None):
    #         if not self.is_fitted():
    #             raise Exception(f'Mock Classifier is not fitted')
    #         else:
    #             return np.array(X) * self.ctr
    #
    #     def predict(self, X):
    #         return np.round(self.predict_proba(X), 0)
    #
    #
    #     def predict_proba(self, X):
    #         if not self._is_fitted:
    #             raise ValueError(f'This instance of Mock CLF is not fitted yet.')
    #         else:
    #             p_0 = np.random.uniform(0, 1, X.shape[0])
    #             p_1 = 1 - p_0
    #             out = np.empty((X.shape[0], 2))
    #             out[:, 0] = p_0
    #             out[:, 1] = p_1
    #             return out
    #
    #
    # return MockCLF   # (ctr=0, sleep=0.2)






























