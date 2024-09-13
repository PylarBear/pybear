# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# this checks the general functionality of the AutoGSTCV module.


from model_selection import AutoGSTCV

import time
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification





X, y = make_classification(
    n_samples=100, n_features=5,
    n_repeated=0, n_informative=5, n_redundant=0
)




agstcv = AutoGSTCV(
    estimator=LogisticRegression(
        max_iter=100, fit_intercept=True, tol=1e-6, solver='lbfgs'
    ),
    params={'C': [np.logspace(-5, 5, 11), 11, 'soft_float']},
    total_passes=11,
    total_passes_is_hard=True,
    max_shifts=None,
    agstcv_verbose=False,
    scoring='balanced_accuracy',
    refit=True,
    n_jobs=None,
    return_train_score=False,
    error_score='raise'
)

t0 = time.perf_counter()
agstcv.fit(X, y)
tf = time.perf_counter() - t0

print(agstcv.best_params_)
print(agstcv.best_threshold_)
print(agstcv.score(X, y))
print(f'total time = {round(tf, 1)}')




