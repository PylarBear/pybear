# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# this test gets the run times for sk GSCV & GSTCV, both using logistic,
# with n_jobs set by a context manager, for GS(T)CV and the estimator
# permuting thru all combinations of [None,1,2,3,4]




import joblib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV as GSTCV

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from utilities._benchmarking import time_memory_benchmark as tmb




param_grid = {
    'C': np.logspace(-6, -3, 4),
    'tol': np.logspace(-7, -4, 4)
}


def _est():
    return sk_LogisticRegression(
        C=1e-3,
        tol=1e-6,
        solver='lbfgs',
        n_jobs=None,
        fit_intercept=False,
        max_iter=100
    )


def _gscv(X, y, n_jobs):
    with joblib.parallel_config(backend='multiprocessing', n_jobs=n_jobs):
        __ = sk_GridSearchCV(
            _est(),
            param_grid,
            scoring=['accuracy', 'balanced_accuracy'],
            n_jobs=None,
            error_score='raise',
            refit='accuracy',
            cv=4,
            verbose=0,
            pre_dispatch="1*n_jobs",
            return_train_score=True
        )

        __.fit(X, y)

    return __


def _gstcv(X, y, n_jobs):
    with joblib.parallel_config(backend='multiprocessing', n_jobs=n_jobs):
        __ = GSTCV(
            _est(),
            param_grid,
            thresholds=[0.5],
            scoring=['accuracy', 'balanced_accuracy'],
            n_jobs=None,
            error_score='raise',
            refit='accuracy',
            cv=4,
            verbose=0,
            return_train_score=True
        )

        __.fit(X, y)

    return __


_rows = 50_000
_cols = 100
X = np.random.randint(0, 10, (_rows, _cols))
y = np.random.randint(0, 2, (_rows,))


njobs_settings = (None, 1, 2, 3, 4)

out = tmb(
    *[(f'est_{_njobs}_gscv_{_njobs}', _gscv, [X, y, _njobs], {}) for _njobs in njobs_settings],
    *[(f'est_{_njobs}_gstcv_{_njobs}', _gstcv, [X, y, _njobs], {}) for _njobs in njobs_settings],
    number_of_trials=3,
    rest_time=3,
    verbose=1
)

mean_times = out[0].mean(axis=1).ravel()
gscv_cutoff = len(njobs_settings)
gscv_times = mean_times[:gscv_cutoff]
gstcv_times = mean_times[gscv_cutoff:]

#    # TIME_MEM_HOLDER SHAPE:
    # axis_0 = time, mem
    # axis_1 = number_of_functions
    # axis_2 = number_of_trials

DF_GSCV = pd.DataFrame(
    index=f'CM_' + np.array(list(map(str, njobs_settings))),
    columns=[f'gscv_time']
).fillna('-')
DF_GSTCV = DF_GSCV.copy()
DF_GSTCV.columns = [f'gstcv_time']
for idx, df_idx in enumerate(njobs_settings):
    DF_GSCV.loc[f'CM_' + str(df_idx), f'gscv_time'] = round(gscv_times[idx], 1)
    DF_GSTCV.loc[f'CM_' + str(df_idx), f'gstcv_time'] = round(gstcv_times[idx], 1)


desktop_path = Path.home() / "Desktop"
DF_GSCV.to_csv(desktop_path / "GSCV_times.csv")
DF_GSTCV.to_csv(desktop_path / "GSTCV_times.csv")











