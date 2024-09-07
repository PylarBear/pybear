# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# this checks the general functionality of the AutoGSTCVDask module.


from model_selection import AutoGSTCVDask

import time
import numpy as np

from dask_ml.linear_model import LogisticRegression
from distributed import Client
from dask_ml.datasets import make_classification





X, y = make_classification(
    n_samples=100, n_features=5,
    n_repeated=0, n_informative=5,
    n_redundant=0, chunks=(100, 5)
)




agstcv = AutoGSTCVDask(
    estimator=LogisticRegression(
        max_iter=100, fit_intercept=True, tol=1e-6, solver='lbfgs'
    ),
    params={'C': [np.logspace(-5, 5, 11), 11, 'soft_float']},
    thresholds=np.linspace(0.1, 0.9, 3),
    total_passes=2,
    total_passes_is_hard=True,
    max_shifts=None,
    agscv_verbose=False,
    scoring='balanced_accuracy',
    refit=True,
    n_jobs=None,
    return_train_score=False,
    error_score='raise',
    cache_cv=True,
    iid=True,
    scheduler=None
)




if __name__ == '__main__':

    with Client(n_workers=None, threads_per_worker=1):

        t0 = time.perf_counter()
        agstcv.fit(X, y)
        tf = time.perf_counter() - t0

        print(agstcv.best_params_)
        print(agstcv.best_threshold_)
        print(agstcv.score(X, y))
        print(f'total time = {round(tf, 1)}')




