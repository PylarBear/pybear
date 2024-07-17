# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.model_selection import KFold



from model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit
from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder import \
    _cv_results_builder


# def _core_fit(
#     _X: XSKWIPType,
#     _y: YSKWIPType,
#     _estimator: ClassifierProtocol,
#     _cv_results: CVResultsType,
#     _cv: Union[int, GenericKFoldType],
#     _error_score: Union[int, float, Literal['raise']],
#     _verbose: int,
#     _scorer: ScorerWIPType,
#     _n_jobs: Union[int, None],
#     _return_train_score: bool,
#     _PARAM_GRID_KEY: npt.NDArray[np.uint8],
#     _THRESHOLD_DICT: dict[int, npt.NDArray[np.float64]],
#     **params
#     ) -> CVResultsType

_PARAM_GRID = [
    {
        'C': np.logspace(-3,3,7),
        'solver': ['saga', 'lbfgs']
    },
    {
        'C': np.logspace(5, 8, 4),
        'solver': ['saga', 'lbfgs']
    },
]

_THRESHOLD_DICT = {
    0: np.linspace(0,1,21),
    1: np.linspace(0,1,11)
}

_n_splits = 5
_verbose = 0
_n_jobs = 4
_return_train_score = True
_SCORER = {
    'accuracy':accuracy_score,
    'balanced_accuracy': balanced_accuracy_score
}
_error_score = 'raise'


_X, _y = make_classification(
    n_classes=2, n_features=5, n_redundant=0,
    n_informative=5, n_samples=100, n_repeated=0
)

_cv = KFold(n_splits=_n_splits).split(_X, _y)

_estimator = LogisticRegression()

_cv_results, _PARAM_GRID_KEY = \
    _cv_results_builder(
        _PARAM_GRID,
        _n_splits,
        _SCORER,
        _return_train_score
    )



out_cv_results =  _core_fit(
    _X,
    _y,
    _estimator,
    _cv_results,
    _cv,
    _error_score,
    _verbose,
    _SCORER,
    _n_jobs,
    _return_train_score,
    _PARAM_GRID_KEY,
    _THRESHOLD_DICT,
)


print(out_cv_results)







