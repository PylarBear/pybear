# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# this checks the general functionality of the AutoGridSearchCVDask module.



from model_selection import AutoGridSearchCVDask

import time
import numpy as np

from dask_ml.linear_model import LogisticRegression
from distributed import Client
from dask_ml.datasets import make_classification



X, y = make_classification(
    n_samples=100, n_features=5,
    n_repeated=0, n_informative=5,
    n_redundant=0, chunks=(100,5)
)




agscv = AutoGridSearchCVDask(
    estimator=LogisticRegression(
        max_iter=100, fit_intercept=True, tol=1e-6, solver='lbfgs'
    ),
    params={'C': [np.logspace(-5, 5, 11), 11, 'soft_float']},
    total_passes=3,
    total_passes_is_hard=True,
    max_shifts=None,
    agscv_verbose=False,
    scoring='balanced_accuracy',
    refit=True,
    n_jobs=4,
    return_train_score=False,
    error_score='raise'
)

if __name__ == '__main__':

    with Client(n_workers=None, threads_per_worker=1):

        t0 = time.perf_counter()
        agscv.fit(X, y)
        tf = time.perf_counter() - t0

        print(agscv.best_params_)
        print(agscv.score(X, y))
        print(f'total time = {round(tf, 1)}')




