# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification


estimator = LogisticRegression(
    penalty="l2",
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="lbfgs",
    max_iter=100,
    multi_class="auto",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)

AutoGridSearch = autogridsearch_wrapper(sklearn_GridSearchCV)

gscv = AutoGridSearch(
    estimator = estimator,
    total_passes=3,
    total_passes_is_hard=False,
    max_shifts=2
)

X, y = make_classification(n_samples=20, n_features=5, n_redundant=0,
                           n_informative=5, n_repeated=0, weights=[0.8, 0.2])
































