# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import dask.array as da
import dask.dataframe as ddf
from uuid import uuid4

from distributed import Client

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import \
    GSTCVDask as dask_GSTCV



@pytest.fixture(scope='session')
def _client():
    client = Client(n_workers=1, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture(scope='session')
def standard_cache_cv():
    return True


@pytest.fixture(scope='session')
def standard_iid():
    return True



# data objects ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.fixture(scope='session')
def _rows():
    return 100


@pytest.fixture(scope='session')
def _cols():
    return 3


@pytest.fixture(scope='session')
def X_da(_rows, _cols):
    da.random.seed(19)
    __ = da.random.randint(0, 10, (_rows, _cols))
    __ = __.rechunk((_rows//10, _cols))
    __ = __.astype(np.float64)  # must be here for dask standard scaler to work
    return __


@pytest.fixture(scope='session')
def X_np(X_da):
    return X_da.compute()


@pytest.fixture(scope='session')
def COLUMNS(_cols):
    return [str(uuid4())[:4] for _ in range(_cols)]


@pytest.fixture(scope='session')
def X_ddf(X_da, COLUMNS):
    return ddf.from_dask_array(X_da, columns=COLUMNS)


@pytest.fixture(scope='session')
def X_pd(X_ddf):
    return X_ddf.compute()


@pytest.fixture(scope='session')
def y_da(_rows):
    np.random.seed(19)
    return da.random.randint(0, 2, (_rows,)).rechunk((_rows/10,))


@pytest.fixture(scope='session')
def y_np(y_da):
    return y_da.compute()


@pytest.fixture(scope='session')
def y_ddf(y_da):
    return ddf.from_dask_array(y_da, columns=['y'])


@pytest.fixture(scope='session')
def y_pd(y_ddf):
    return y_ddf.compute()

# END data objects ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# @pytest.fixture(scope='session')
# def dask_log_init_params():
#
#     return {
#         'C':1e-8,
#         'tol': 1e-1, # need 1e-6 here to pass accuracy est accuracy tests
#         'max_iter': 2, # need 10000 here to pass accuracy est accuracy tests
#         'fit_intercept': False,
#         'solver': 'lbfgs',
#         'random_state': 69
#     }


# END estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** *

# transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def dask_est_log(sk_log_init_params):
    # 25_04_29 converted this to sklearn because...
    # 1) to speed up tests
    # 2) dask_ml KFold & LogisticRegression expressly block ddfs
    # 3) dask_ml is always breaking
    return sk_LogisticRegression(**sk_log_init_params)

# END transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** *


# est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def param_grid_dask_log():
    return {'C': [1e-4, 1e-5]}

# END est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gstcv init params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def dask_gstcv_init_params(
    dask_est_log, param_grid_dask_log, standard_cv_int, standard_error_score
):
    return {
        'estimator': dask_est_log,
        'param_grid': param_grid_dask_log,
        'thresholds': [0.4,0.5,0.6],
        'scoring': 'accuracy',
        'n_jobs': 1,
        'refit': False,
        'cv': standard_cv_int,
        'verbose': 0,
        'error_score': standard_error_score,
        'return_train_score': False,
        'iid': True,
        'cache_cv': True,
        'scheduler': None
    }



# END gs(t)cv init params ** * ** * ** * ** * ** * ** * ** * ** * ** *









# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gstcv log est one scorer, various refits
@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_prefit(
    dask_gstcv_init_params, dask_est_log, param_grid_dask_log
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_est_log,
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da(
    dask_gstcv_init_params, dask_est_log, param_grid_dask_log, X_da,
    y_da#, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_est_log,
        param_grid=param_grid_dask_log,
        refit='accuracy'
    )
    return __.fit(X_da, y_da)

# END gstcv log est one scorer, various refits

# END ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gstcv log est two scorers, various refits

@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da(
    dask_gstcv_init_params, dask_est_log, param_grid_dask_log, X_da,
    y_da#, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_est_log,
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_da, y_da)

# END gstcv log est two scorers, various refits

# END ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** *
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *







