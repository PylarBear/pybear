# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this test gets the run times for sk GSCV & GSTCV, both using logistic,
# with n_jobs for GS(T)CV and the estimator permuting thru all combinations
# of [None,1,2,3,4]
#
# GSTCV:
#     	    gstcv_None	gstcv_1	gstcv_2	gstcv_3	gstcv_4
#  est_None	    4.3	    4.1	    13.8	15.5	13.2
#     est_1	    4	    4	    13.8	15.4	12.9
#     est_2	    72.8	71.4	13.3	18.2	19.4
#     est_3	    72.2	71.7	15.8	15.7	15.1
#     est_4	    71.1	71.9	16.5	15.7	14.6
#
# GSCV:
#             gscv_None	gscv_1	gscv_2	gscv_3	gscv_4
#  est_None	    3.7	    3.4	    4.5 	4	    2.5
#     est_1	    3.4	    3.5	    4.1	    3.9	    2.5
#     est_2	    73.3	72.7	3.7	    5.8	    6.2
#     est_3	    71.6	70.8	6.1	    3.9	    3.8
#     est_4	    71	    70.7	6.4	    3.9	    3.2


import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV as GSTCV
from pybear.utilities._benchmarking import time_memory_benchmark as tmb




param_grid = {
    'C': np.logspace(-6, -3, 4),
    'tol': np.logspace(-7, -4, 4)
}


def _est(njobs=1):
    return sk_LogisticRegression(
        C=1e-3,
        tol=1e-6,
        solver='lbfgs',
        n_jobs=njobs,
        fit_intercept=False,
        max_iter=100
    )


def _gscv(X, y, est_njobs=1, gscv_njobs=1):
    return sk_GridSearchCV(
        _est(est_njobs),
        param_grid,
        scoring=['accuracy', 'balanced_accuracy'],
        n_jobs=gscv_njobs,
        error_score='raise',
        refit='accuracy',
        cv=4,
        verbose=0,
        return_train_score=True
    ).fit(X, y)


def _gstcv(X, y, est_njobs=1, gstcv_njobs=1):
    return GSTCV(
        _est(est_njobs),
        param_grid,
        thresholds=[0.5],
        scoring=['accuracy', 'balanced_accuracy'],
        n_jobs=gstcv_njobs,
        error_score='raise',
        refit='accuracy',
        cv=4,
        verbose=0,
        return_train_score=True
    ).fit(X, y)


_rows = 10_000
_cols = 100
X = np.random.randint(0, 10, (_rows, _cols))
y = np.random.randint(0, 2, (_rows,))


njobs_settings = (None, 1, 2, 3, 4)
pairs = list(itertools.product(njobs_settings, njobs_settings))

out = tmb(
    # *[(f'est_{est_njobs}_gscv_{gscv_njobs}', _gscv, [X, y, est_njobs, gscv_njobs], {}) for est_njobs, gscv_njobs in pairs],
    *[(f'est_{est_njobs}_gstcv_{gscv_njobs}', _gstcv, [X, y, est_njobs, gscv_njobs], {}) for est_njobs, gscv_njobs in pairs],
    number_of_trials=2,
    rest_time=1,
    verbose=1
)

mean_times = out[0].mean(axis=1).ravel()
gscv_cutoff = len(pairs)
# gscv_times = mean_times[:gscv_cutoff]
gstcv_times = mean_times[:gscv_cutoff]

#    # TIME_MEM_HOLDER SHAPE:
    # axis_0 = time, mem
    # axis_1 = number_of_functions
    # axis_2 = number_of_trials

DF_GSCV = pd.DataFrame(
    index=f'est_' + np.array(list(map(str, njobs_settings))),
    columns=f'gscv_' + np.array(list(map(str, njobs_settings)))
).fillna('-')
DF_GSTCV = DF_GSCV.copy()
DF_GSTCV.columns = f'gstcv_' + np.array(list(map(str, njobs_settings)))
for idx, (df_idx, df_col) in enumerate(pairs):
    # DF_GSCV.loc[f'est_' + str(df_idx), f'gscv_' + str(df_col)] = round(gscv_times[idx], 1)
    DF_GSTCV.loc[f'est_' + str(df_idx), f'gstcv_' + str(df_col)] = round(gstcv_times[idx], 1)


desktop_path = Path.home() / "Desktop"
# DF_GSCV.to_csv(desktop_path / "GSCV_times.csv")
DF_GSTCV.to_csv(desktop_path / "GSTCV_times.csv")











