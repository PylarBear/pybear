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

from dask_ml.preprocessing import StandardScaler as dask_StandardScaler

from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
# from xgboost.dask import DaskXGBClassifier as dask_XGBClassifier
from xgboost import XGBClassifier as dask_XGBClassifier

from sklearn.pipeline import Pipeline

from dask_ml.model_selection import GridSearchCV as dask_GSCV

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask as dask_GSTCV




@pytest.fixture(scope='session')
def _client():
    client = Client(n_workers=None, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture(scope='session')
def standard_cache_cv():
    return True


@pytest.fixture(scope='session')
def standard_iid():
    return True


#
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

@pytest.fixture(scope='session')
def dask_log_init_params():

    return {
        'C':1e-8,
        'tol': 1e-1, # need 1e-6 here to pass accuracy est/pipe accuracy tests
        'max_iter': 2, # need 10000 here to pass accuracy est/pipe accuracy tests
        'fit_intercept': False,
        'solver': 'newton',
        'random_state': 69
    }
    #         return dask_LogisticRegression(
    #             max_iter=10_000,
    #             solver='newton',
    #             random_state=69,
    #             tol=1e-6


@pytest.fixture(scope='session')
def dask_xgb_init_params():
    return {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.01,
        'tree_method': 'hist'
    }

# END estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** *

# transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.fixture(scope='session')
def dask_standard_scaler():
    # as of 24_08_26, the only way to get repeatable results with dask
    # StandardScaler is with_mean & with_std both False. under circum-
    # stances when not both False, not getting the exact same output
    # given the same data. this compromises _core_fit accuracy tests.
    return dask_StandardScaler(with_mean=False, with_std=False)


@pytest.fixture(scope='session')
def dask_est_log(dask_log_init_params):
    return dask_LogisticRegression(**dask_log_init_params)


@pytest.fixture(scope='session')
def dask_est_xgb(dask_xgb_init_params):
    return dask_XGBClassifier(**dask_xgb_init_params)

# END transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** *


# est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def param_grid_dask_log():
    return {'C': [1e-4, 1e-5]}


@pytest.fixture(scope='session')
def param_grid_dask_xgb():
    return {'max_depth': [4, 5]}

# END est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gs(t)cv init params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def dask_gscv_init_params(
    dask_log_init_params, param_grid_dask_log, standard_cv_int, standard_error_score
):

    return {
        'estimator': dask_LogisticRegression(),
        'param_grid': {},
        'scoring': 'accuracy',
        'n_jobs': -1,
        'refit': False,
        'cv': standard_cv_int,
        'error_score': standard_error_score,
        'return_train_score': False,
        'iid': True,
        'cache_cv': True,
        'scheduler': None
    }


@pytest.fixture(scope='session')
def dask_gstcv_init_params(
    dask_log_init_params, param_grid_dask_log, standard_cv_int, standard_error_score
):
    return {
        'estimator': dask_LogisticRegression(),
        'param_grid': {},
        'thresholds': [0.4,0.5,0.6],
        'scoring': 'accuracy',
        'n_jobs': None,
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

# gscv log est one scorer, various refits
@pytest.fixture(scope='session')
def dask_GSCV_est_log_one_scorer_prefit(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSCV_est_log_one_scorer_postfit_refit_false(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da, y_da,
    _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_est_log_one_scorer_postfit_refit_str(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da, y_da,
    _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_est_log_one_scorer_postfit_refit_fxn(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da, y_da,
    _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)

# END gscv log est one scorer, various refits



# gstcv log est one scorer, various refits
@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_prefit(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da, y_da,
    _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=False
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit='accuracy'
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_ddf, y_ddf)

# END gstcv log est one scorer, various refits



# gscv xgb est one scorer, various refits
# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_one_scorer_prefit(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_one_scorer_postfit_refit_false(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_one_scorer_postfit_refit_str(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_one_scorer_postfit_refit_fxn(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gscv xgb est one scorer, various refits



# gstcv xgb est one scorer, various refits
# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_one_scorer_prefit(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_one_scorer_postfit_refit_false(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_one_scorer_postfit_refit_str(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_one_scorer_postfit_refit_fxn(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gstcv xgb est one scorer, various refits

# END ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gscv log est two scorers, various refits
@pytest.fixture(scope='session')
def dask_GSCV_est_log_two_scorers_prefit(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSCV_est_log_two_scorers_postfit_refit_false(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_est_log_two_scorers_postfit_refit_str(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_est_log_two_scorers_postfit_refit_fxn(
    dask_gscv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)

# END gscv log est two scorers, various refits



# gstcv log est two scorers, various refits
@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_prefit(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log
):

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_da,
    y_da, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, param_grid_dask_log, X_ddf,
    y_ddf, _client
):
    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=dask_LogisticRegression(**dask_log_init_params),
        param_grid=param_grid_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_ddf, y_ddf)

# END gstcv log est two scorers, various refits



# gscv xgb est two scorers, various refits
# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_two_scorers_prefit(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb
# ):
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_two_scorers_postfit_refit_false(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_two_scorers_postfit_refit_str(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_est_xgb_two_scorers_postfit_refit_fxn(
#     dask_gscv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gscv xgb est two scorers, various refits



# gstcv xgb est two scorers, various refits
# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_two_scorers_prefit(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb
# ):
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_two_scorers_postfit_refit_false(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_two_scorers_postfit_refit_str(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_est_xgb_two_scorers_postfit_refit_fxn(
#     dask_gstcv_init_params, dask_xgb_init_params, param_grid_dask_xgb, X_da,
#     y_da, _client
# ):
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=dask_XGBClassifier(**dask_xgb_init_params),
#         param_grid=param_grid_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gstcv xgb est two scorers, various refits

# END ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** *
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *





# pipeline esimators ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def dask_pipe_log(dask_standard_scaler, dask_est_log):
    return Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_est_log)
        ]
    )


# @pytest.fixture(scope='session')
# def dask_pipe_xgb(dask_standard_scaler, dask_est_xgb):
#     return Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_est_xgb)
#         ]
#     )

# END pipeline esimators ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def param_grid_pipe_dask_log():
    return {
        'dask_StandardScaler__with_mean': [True, False],
        'dask_logistic__C': [1e-4, 1e-5]
    }


@pytest.fixture(scope='session')
def param_grid_pipe_dask_xgb():
    return {
        'dask_StandardScaler__with_mean': [True, False],
        'dask_XGB__max_depth': [4, 5]
    }

# END pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log pipe one scorer, various refits
@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_one_scorer_prefit(
    dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_one_scorer_postfit_refit_false(
    dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_one_scorer_postfit_refit_str(
    dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_one_scorer_postfit_refit_fxn(
    dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)

# END gscv log pipe one scorer, various refits



# gstcv log pipe one scorer, various refits
@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_prefit(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=False
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit='accuracy'
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        refit=lambda x: 0
    )
    return __.fit(X_ddf, y_ddf)

# END gstcv log pipe one scorer, various refits



# gscv xgb pipe one scorer, various refits
# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_one_scorer_prefit(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_one_scorer_postfit_refit_false(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_one_scorer_postfit_refit_str(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_one_scorer_postfit_refit_fxn(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gscv xgb pipe one scorer, various refits



# gstcv xgb pipe one scorer, various refits
# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_one_scorer_prefit(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_one_scorer_postfit_refit_false(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_one_scorer_postfit_refit_str(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_one_scorer_postfit_refit_fxn(
#         dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#         param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gstcv xgb pipe one scorer, various refits

# END PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# PIPELINES - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log pipe two scorers, various refits
@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_two_scorers_prefit(
        dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_two_scorers_postfit_refit_false(
    dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_two_scorers_postfit_refit_str(
        dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSCV_pipe_log_two_scorers_postfit_refit_fxn(
        dask_gscv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSCV(**dask_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)

# END gscv log pipe two scorers, various refits


# gstcv log pipe two scorers, various refits
@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_prefit(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_da(
        dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_ddf(
        dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_da(
        dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
        param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_ddf, y_ddf)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_da(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_da, y_da, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_da, y_da)


@pytest.fixture(scope='session')
def dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_ddf(
    dask_gstcv_init_params, dask_log_init_params, dask_standard_scaler,
    param_grid_pipe_dask_log, X_ddf, y_ddf, _client
):

    pipe = Pipeline(
        steps=[
            ('dask_StandardScaler', dask_standard_scaler),
            ('dask_logistic', dask_LogisticRegression(**dask_log_init_params))
        ]
    )

    __ = dask_GSTCV(**dask_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_dask_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_ddf, y_ddf)

# END gstcv log pipe two scorers, various refits



# gscv xgb pipe two scorers, various refits
# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_two_scorers_prefit(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_two_scorers_postfit_refit_false(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_two_scorers_postfit_refit_str(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSCV_pipe_xgb_two_scorers_postfit_refit_fxn(
#     dask_gscv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSCV(**dask_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gscv xgb pipe two scorers, various refits



# gstcv xgb pipe two scorers, various refits
# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_two_scorers_prefit(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_two_scorers_postfit_refit_false(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_two_scorers_postfit_refit_str(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_da, y_da)


# @pytest.fixture(scope='session')
# def dask_GSTCV_pipe_xgb_two_scorers_postfit_refit_fxn(
#     dask_gstcv_init_params, dask_xgb_init_params, dask_standard_scaler,
#     param_grid_pipe_dask_xgb, X_da, y_da, _client
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('dask_StandardScaler', dask_standard_scaler),
#             ('dask_XGB', dask_XGBClassifier(**dask_xgb_init_params))
#         ]
#     )
#
#     __ = dask_GSTCV(**dask_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_dask_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_da, y_da)

# END gstcv xgb pipe two scorers, various refits

# END PIPELINES - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


































