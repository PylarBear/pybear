# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression

from xgboost import XGBClassifier as sk_XGBClassifier
from xgboost.dask import DaskXGBClassifier as dask_XGBClassifier



def make_est__param_grid():

    params = { 'max_iter': 100, 'tol': 1e-4, 'solver': 'lbfgs',
        'fit_intercept': False}
    sk_clf = sk_LogisticRegression(**params)
    dask_clf = dask_LogisticRegression(**params)
    _param_grid = {'C': [1e-5, 1e-4, 1e-3]}

    # xgb
    # params = {'n_estimators': 50, 'max_depth': 5, 'tree_method': 'hist'}
    # sk_clf = sk_XGBClassifier(**params)
    # dask_clf = sk_XGBClassifier(**params)
    # _param_grid = {'max_depth': [3,4,5]}  # xgb


    return sk_clf, dask_clf, _param_grid