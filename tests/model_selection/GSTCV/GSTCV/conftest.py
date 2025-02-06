# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
from uuid import uuid4

from sklearn.preprocessing import StandardScaler as sk_StandardScaler

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from xgboost import XGBClassifier as sk_XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV as sk_GSCV

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV as sk_GSTCV



#
# data objects ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.fixture(scope='session')
def _rows():
    return 100


@pytest.fixture(scope='session')
def _cols():
    return 3


@pytest.fixture(scope='session')
def X_np(_rows, _cols):
    np.random.seed(19)
    return np.random.randint(0, 10, (_rows, _cols))


@pytest.fixture(scope='session')
def COLUMNS(_cols):
    return [str(uuid4())[:4] for _ in range(_cols)]



@pytest.fixture(scope='session')
def X_pd(X_np, COLUMNS):
    return pd.DataFrame(data=X_np, columns=COLUMNS)



@pytest.fixture(scope='session')
def y_np(_rows):
    np.random.seed(19)
    return np.random.randint(0, 2, (_rows,))


@pytest.fixture(scope='session')
def y_pd(y_np):
    return pd.DataFrame(data=y_np, columns=['y'])

# END data objects ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def sk_log_init_params():
    return {
        'C':1e-8,
        'tol': 1e-1,
        'max_iter': 1,
        'fit_intercept': False,
        'solver': 'lbfgs'
    }


@pytest.fixture(scope='session')
def sk_xgb_init_params():
    return {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.01,
        'tree_method': 'hist'
    }

# END estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** *

# transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.fixture(scope='session')
def sk_standard_scaler():
    return sk_StandardScaler(with_mean=True, with_std=True)

@pytest.fixture(scope='session')
def sk_est_log(sk_log_init_params):
    return sk_LogisticRegression(**sk_log_init_params)


@pytest.fixture(scope='session')
def sk_est_xgb(sk_xgb_init_params):
    return sk_XGBClassifier(**sk_xgb_init_params)

# END transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** *


# est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def param_grid_sk_log():
    return {'C': [1e-4, 1e-5]}


@pytest.fixture(scope='session')
def param_grid_sk_xgb():
    return {'max_depth': [4, 5]}

# END est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gs(t)cv init params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def sk_gscv_init_params(
    sk_log_init_params, param_grid_sk_log, standard_cv_int, standard_error_score
):

    return {
        'estimator': sk_LogisticRegression(),
        'param_grid': {},
        'scoring': 'accuracy',
        'n_jobs': 1,
        'refit': False,
        'cv': standard_cv_int,
        'verbose': 0,
        'pre_dispatch': '2*n_jobs',
        'error_score': standard_error_score,
        'return_train_score': False
    }


@pytest.fixture(scope='session')
def sk_gstcv_init_params(
    sk_log_init_params, param_grid_sk_log, standard_cv_int, standard_error_score
):
    return {
        'estimator': sk_LogisticRegression(),
        'param_grid': {},
        'thresholds': [0.4,0.5,0.6],
        'scoring': 'accuracy',
        'n_jobs': 1,
        'refit': False,
        'cv': standard_cv_int,
        'verbose': 0,
        'error_score': standard_error_score,
        'return_train_score': False
    }



# END gs(t)cv init params ** * ** * ** * ** * ** * ** * ** * ** * ** *









# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log est one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_prefit(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_false(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_str(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_fxn(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log est one scorer, various refits



# gstcv log est one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_prefit(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log est one scorer, various refits



# gscv xgb est one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSCV_est_xgb_one_scorer_prefit(
    sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        refit=False
    )
    return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_one_scorer_postfit_refit_false(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_one_scorer_postfit_refit_str(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_one_scorer_postfit_refit_fxn(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gscv xgb est one scorer, various refits



# gstcv xgb est one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_est_xgb_one_scorer_prefit(
    sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_est_xgb_one_scorer_postfit_refit_false(
    sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_xgb_one_scorer_postfit_refit_str(
    sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_xgb_one_scorer_postfit_refit_fxn(
    sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gstcv xgb est one scorer, various refits

# END ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gscv log est two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_prefit(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_false(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_str(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_fxn(
    sk_gscv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log est two scorers, various refits



# gstcv log est two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_prefit(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log
):

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_LogisticRegression(**sk_log_init_params),
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log est two scorers, various refits



# gscv xgb est two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSCV_est_xgb_two_scorers_prefit(
    sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb
):

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_XGBClassifier(**sk_xgb_init_params),
        param_grid=param_grid_sk_xgb,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_two_scorers_postfit_refit_false(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_two_scorers_postfit_refit_str(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_est_xgb_two_scorers_postfit_refit_fxn(
#     sk_gscv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gscv xgb est two scorers, various refits


# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# gstcv xgb est two scorers, various refits
# @pytest.fixture(scope='session')
# def sk_GSTCV_est_xgb_two_scorers_prefit(
#     sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb
# ):
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_est_xgb_two_scorers_postfit_refit_false(
#     sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_est_xgb_two_scorers_postfit_refit_str(
#     sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_est_xgb_two_scorers_postfit_refit_fxn(
#     sk_gstcv_init_params, sk_xgb_init_params, param_grid_sk_xgb, X_np, y_np
# ):
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=sk_XGBClassifier(**sk_xgb_init_params),
#         param_grid=param_grid_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gstcv xgb est two scorers, various refits

# END ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** *
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *





# pipeline estimators ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def sk_pipe_log(sk_standard_scaler, sk_est_log):
    return Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
        ]
    )


@pytest.fixture(scope='session')
def sk_pipe_xgb(sk_standard_scaler, sk_est_xgb):
    return Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_XGB', sk_est_xgb)
        ]
    )

# END pipeline esimators ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def param_grid_pipe_sk_log():
    return {
        'sk_StandardScaler__with_mean': [True, False],
        'sk_logistic__C': [1e-4, 1e-5]
    }


@pytest.fixture(scope='session')
def param_grid_pipe_sk_xgb():
    return {
        'sk_StandardScaler__with_mean': [True, False],
        'sk_XGB__max_depth': [4, 5]
    }

# END pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log pipe one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_one_scorer_prefit(
    sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_one_scorer_postfit_refit_false(
    sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_one_scorer_postfit_refit_str(
    sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_one_scorer_postfit_refit_fxn(
    sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log pipe one scorer, various refits



# gstcv log pipe one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_prefit(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log pipe one scorer, various refits



# gscv xgb pipe one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSCV_pipe_xgb_one_scorer_prefit(
    sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
    param_grid_pipe_sk_xgb
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_xgb,
        refit=False
    )
    return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_one_scorer_postfit_refit_false(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_one_scorer_postfit_refit_str(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_one_scorer_postfit_refit_fxn(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gscv xgb pipe one scorer, various refits


# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# gstcv xgb pipe one scorer, various refits
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_one_scorer_prefit(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit=False
#     )
#     return __


# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_one_scorer_postfit_refit_false(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_one_scorer_postfit_refit_str(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_one_scorer_postfit_refit_fxn(
#         sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#         param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gstcv xgb pipe one scorer, various refits

# END PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# PIPELINES - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log pipe two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_two_scorers_prefit(
        sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_two_scorers_postfit_refit_false(
    sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_two_scorers_postfit_refit_str(
        sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_two_scorers_postfit_refit_fxn(
        sk_gscv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log pipe two scorers, various refits


# gstcv log pipe two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_prefit(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_np(
        sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_pd(
        sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_np(
        sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
        param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_log_init_params, sk_standard_scaler,
    param_grid_pipe_sk_log, X_pd, y_pd
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_LogisticRegression(**sk_log_init_params))
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log pipe two scorers, various refits


# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# gscv xgb pipe two scorers, various refits
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_two_scorers_prefit(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_two_scorers_postfit_refit_false(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_two_scorers_postfit_refit_str(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSCV_pipe_xgb_two_scorers_postfit_refit_fxn(
#     sk_gscv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSCV(**sk_gscv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gscv xgb pipe two scorers, various refits


# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# gstcv xgb pipe two scorers, various refits
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_two_scorers_prefit(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_two_scorers_postfit_refit_false(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=False
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_two_scorers_postfit_refit_str(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit='accuracy'
#     )
#     return __.fit(X_np, y_np)

# xfail isnt catching this
# @pytest.mark.xfail(reason=f"pizza need xgboost update for sklearn==1.6")
# @pytest.fixture(scope='session')
# def sk_GSTCV_pipe_xgb_two_scorers_postfit_refit_fxn(
#     sk_gstcv_init_params, sk_xgb_init_params, sk_standard_scaler,
#     param_grid_pipe_sk_xgb, X_np, y_np
# ):
#
#     pipe = Pipeline(
#         steps=[
#             ('sk_StandardScaler', sk_standard_scaler),
#             ('sk_XGB', sk_XGBClassifier(**sk_xgb_init_params))
#         ]
#     )
#
#     __ = sk_GSTCV(**sk_gstcv_init_params)
#     __.set_params(
#         estimator=pipe,
#         param_grid=param_grid_pipe_sk_xgb,
#         scoring=['accuracy', 'balanced_accuracy'],
#         refit=lambda x: 0
#     )
#     return __.fit(X_np, y_np)

# END gstcv xgb pipe two scorers, various refits

# END PIPELINES - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


































