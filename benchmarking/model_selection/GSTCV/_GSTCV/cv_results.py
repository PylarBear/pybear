# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this module lets you inspect how cv_results_ looks


import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.model_selection import KFold

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from pybear.model_selection.GSTCV._GSTCVMixin._fit._cv_results._cv_results_builder \
    import _cv_results_builder



_X, _y = make_classification(
    n_classes=2, n_features=10, n_redundant=0,
    n_informative=5, n_samples=200, n_repeated=0
)

_PARAM_GRID = [
    {'C': np.logspace(-3, 3, 3), 'solver': ['saga', 'lbfgs']},
    {'C': np.logspace(4, 8, 3), 'solver': ['saga', 'lbfgs']}
]

_n_splits = 5

_SCORER = {
    'accuracy':accuracy_score,
    'balanced_accuracy': balanced_accuracy_score
}

_GSTCV = GSTCV(
    estimator=LogisticRegression(max_iter=10_000),
    param_grid=_PARAM_GRID,
    thresholds=np.linspace(0,1,11),
    scoring=_SCORER,
    n_jobs=4,
    refit=False,
    cv=KFold(n_splits=_n_splits).split(_X, _y),
    verbose=0,
    pre_dispatch='2*n_jobs',
    error_score='raise',
    return_train_score=True
)

# _cv_results, _PARAM_GRID_KEY = \
#     _cv_results_builder(
#         _PARAM_GRID,
#         _n_splits,
#         _SCORER,
#         _return_train_score=True
# )



out_cv_results = pd.DataFrame(_GSTCV.fit(_X, _y).cv_results_)


print(out_cv_results.T)    # notice the T







