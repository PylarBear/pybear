# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LogisticRegression as sklearn_Logistic

import dask.array as da
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_Logistic
import distributed

from pybear.model_selection import GSTCV, GSTCVDask
from pybear.utilities._benchmarking import time_memory_benchmark as tmb

# ----------------------------
# 24_8_10 LINUX, size=1e7, cv=5, chunks=5
# NO PERSISTENCE OF X & Y IN _parallelized_{fit/scorer/train_scorer}
# skGSCV_1e-4      time = 10.665 +/- 0.123 sec; mem = 110.250 +/- 0.433 MB
# skGSTCV_1e-4     time = 12.242 +/- 0.238 sec; mem = 0.000 +/- 0.000 MB
# daGSCV_1e-4      time = 21.540 +/- 5.421 sec; mem = -136.750 +/- 0.433 MB
# daGSTCV_1e-4     time = 164.381 +/- 1.416 sec; mem = 38.000 +/- 1.225 MB

# 24_8_10 LINUX, size=1e7, cv=5, chunks=5
# PERSIST X & Y IN _parallelized_{fit/scorer/train_scorer}
# skGSCV_1e-4      time = 10.040 +/- 0.122 sec; mem = 109.000 +/- 1.000 MB
# skGSTCV_1e-4     time = 12.532 +/- 1.347 sec; mem = 23.000 +/- 39.262 MB
# daGSCV_1e-4      time = 23.609 +/- 4.318 sec; mem = -146.000 +/- 9.670 MB
# daGSTCV_1e-4     time = 74.573 +/- 1.353 sec; mem = 34.750 +/- 0.433 MB

# 24_8_10 LINUX, size=1e7, cv=5, chunks=5
# PERSIST ONLY Y IN _parallelized_{fit/scorer/train_scorer}
# skGSCV_1e-4      time = 10.722 +/- 0.077 sec; mem = 108.500 +/- 0.866 MB
# skGSTCV_1e-4     time = 12.540 +/- 0.550 sec; mem = 45.750 +/- 45.751 MB
# daGSCV_1e-4      time = 17.588 +/- 0.134 sec; mem = -162.500 +/- 38.082 MB
# daGSTCV_1e-4     time = 203.581 +/- 3.183 sec; mem = 38.750 +/- 1.785 MB

# 24_8_10 LINUX, size=1e7, cv=5, chunks=1
# PERSIST X & Y IN _parallelized_{fit/scorer/train_scorer}
# skGSCV_1e-4      time = 10.575 +/- 0.577 sec; mem = 122.250 +/- 4.206 MB
# skGSTCV_1e-4     time = 13.090 +/- 2.156 sec; mem = 0.000 +/- 0.000 MB
# daGSCV_1e-4      time = 23.481 +/- 3.735 sec; mem = -147.250 +/- 12.930 MB
# daGSTCV_1e-4     time = 41.533 +/- 0.320 sec; mem = 14.500 +/- 6.727 MB

# ----------------------------


X_da = da.random.randint(0, 10, (1_000_000, 5)).rechunk((1_000_000, 5))
y_da = da.random.randint(0, 2, (1_000_000,)).rechunk((1_000_000,))
X_np = X_da.compute()
y_np = y_da.compute()


logistic_params = {
    'max_iter': 10000,
    'tol': 1e-6,
    'fit_intercept': False,
    'n_jobs': None,
    'solver': 'lbfgs'
}


def skGSCV(X, y, C=10):
    skGSCV = sklearn_GridSearchCV(
        estimator=sklearn_Logistic(**logistic_params),
        param_grid={'C':[C]},
        cv=5,
        scoring=['balanced_accuracy','accuracy'],
        refit='balanced_accuracy',
        return_train_score=True,
        n_jobs=-1
    )

    skGSCV.fit(X,y)

    return skGSCV


def skGSTCV(X, y, C=10):
    skGSTCV = GSTCV(
        estimator=sklearn_Logistic(**logistic_params),
        param_grid={'C': [C]},
        cv=5,
        scoring=['balanced_accuracy', 'accuracy'],
        refit='balanced_accuracy',
        thresholds=[0.5],
        return_train_score=True,
        n_jobs=-1,
        verbose=0
    )

    skGSTCV.fit(X, y)

    return skGSTCV


def daGSCV(X, y, C=10):
    daGSCV = dask_GridSearchCV(
        estimator=dask_Logistic(**logistic_params),
        param_grid={'C':[C]},
        cv=5,
        scoring=['balanced_accuracy','accuracy'],
        refit='balanced_accuracy',
        return_train_score=True,
        scheduler=None
    )

    daGSCV.fit(X,y)

    return daGSCV


def daGSTCV(X, y, C=10):
    daGSTCV = GSTCVDask(
        estimator=dask_Logistic(**logistic_params),
        param_grid={'C': [C]},
        cv=5,
        scoring=['balanced_accuracy', 'accuracy'],
        refit='balanced_accuracy',
        thresholds=[0.5],
        return_train_score=True,
        scheduler=None,
        verbose=0
    )

    daGSTCV.fit(X, y)

    return daGSTCV




if __name__ == '__main__':

    # with distributed.Client(n_workers=4, threads_per_worker=1) as scheduler:
    #
    #     print(f'dashboard url = ', scheduler.dashboard_link)

    tmb(
        ('skGSCV_1e-4', skGSCV, [X_np, y_np], {'C':1e-4}),
        ('skGSTCV_1e-4', skGSTCV, [X_np, y_np], {'C':1e-4}),
        # ('daGSCV_1e-4', daGSCV, [X_da, y_da], {'C': 1e-4}),
        # ('daGSTCV_1e-4', daGSTCV, [X_da, y_da], {'C': 1e-4}),
        rest_time=3,
        number_of_trials=6
    )











