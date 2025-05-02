# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np
import pandas as pd
from uuid import uuid4

from sklearn.preprocessing import StandardScaler as sk_StandardScaler

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV as sk_GSCV

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV as sk_GSTCV




@pytest.fixture(scope='session')
def standard_cv_int():
    return 4


@pytest.fixture(scope='session')
def standard_error_score():
    return 'raise'




# exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def generic_no_attribute_1():
    def foo(_gscv_type, _attr):
        return f"'{_gscv_type}' object has no attribute '{_attr}'"

    return foo


@pytest.fixture(scope='session')
def generic_no_attribute_2():
    def foo(_gscv_type, _attr):
        return f"This '{_gscv_type}' has no attribute '{_attr}'"

    return foo


# pizza! on the block to go!
# @pytest.fixture(scope='session')
# def generic_no_attribute_3():
#     def foo(_gscv_type, _attr):
#         return f"{_gscv_type} object has no {_attr} attribute."
#
#     return foo


@pytest.fixture(scope='session')
def _no_refit():
    def foo(_object, _apostrophes: bool, _method):
        if _apostrophes:
            __ = "`refit=False`"
        else:
            __ = "refit=False"

        return (f"This {_object} instance was initialized with {__}. "
            f"\n{_method} is available only after refitting on the best "
            f"parameters.")

    return foo


@pytest.fixture(scope='session')
def _refit_false():

    def foo(_gstcv_type):
        return (f"This {_gstcv_type} instance was initialized "
            f"with `refit=False`. \nclasses_ is available only after "
            "refitting on the best parameters."
        )

    return foo



@pytest.fixture(scope='session')
def _not_fitted():
    def foo(_object):
        return (f"This {_object} instance is not fitted yet.\nCall 'fit' "
            f"with appropriate arguments before using this estimator.")

    return foo


@pytest.fixture(scope='session')
def non_num_X():
    return re.escape(f"dtype='numeric' is not compatible with arrays of "
        f"bytes/strings. Convert your data to numeric values explicitly "
        f"instead."
    )


@pytest.fixture(scope='session')
def partial_feature_names_exc():
    return f"The feature names should match those that were passed during fit."


@pytest.fixture(scope='session')
def multilabel_y():
    return re.escape(f"Classification metrics can't handle a mix of "
        f"multilabel-indicator and binary targets")


@pytest.fixture(scope='session')
def non_binary_y():

    def foo(_gstcv_type):
        return re.escape(f"{_gstcv_type} can only perform thresholding on binary "
            f"targets with values in [0,1]. \nPass 'y' as a vector of 0's and 1's.")

    return foo

@pytest.fixture(scope='session')
def different_rows():

    def foo(y_rows, X_rows):
        return re.escape(f"Found input variables with inconsistent "
                         f"numbers of samples: [{y_rows}, {X_rows}]")

    return foo


# END exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *








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

# END estimator init params ** * ** * ** * ** * ** * ** * ** * ** * ** *

# transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.fixture(scope='session')
def sk_standard_scaler():
    return sk_StandardScaler(with_mean=True, with_std=True)

@pytest.fixture(scope='session')
def sk_est_log(sk_log_init_params):
    return sk_LogisticRegression(**sk_log_init_params)

# END transformers / estimators ** * ** * ** * ** * ** * ** * ** * ** *


# est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def param_grid_sk_log():
    return {'C': [1e-4, 1e-5]}

# END est param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gs(t)cv init params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def sk_gscv_init_params(
    sk_est_log, param_grid_sk_log, standard_cv_int, standard_error_score
):

    return {
        'estimator': sk_est_log,
        'param_grid': param_grid_sk_log,
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
    sk_est_log, param_grid_sk_log, standard_cv_int, standard_error_score
):
    return {
        'estimator': sk_est_log,
        'param_grid': param_grid_sk_log,
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
    sk_gscv_init_params, sk_est_log, param_grid_sk_log
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_false(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_str(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_one_scorer_postfit_refit_fxn(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log est one scorer, various refits



# gstcv log est one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_prefit(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log est one scorer, various refits

# END ESTIMATORS - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# ESTIMATORS - TWO SCORERS ** * ** * ** * ** * ** * ** * ** * ** * ** *

# gscv log est two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_prefit(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_false(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_str(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSCV_est_log_two_scorers_postfit_refit_fxn(
    sk_gscv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSCV(**sk_gscv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gscv log est two scorers, various refits



# gstcv log est two scorers, various refits
@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_prefit(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log
):

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=False
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit='accuracy'
    )
    return __.fit(X_pd, y_pd)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_np, y_np
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)


@pytest.fixture(scope='session')
def sk_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_pd(
    sk_gstcv_init_params, sk_est_log, param_grid_sk_log, X_pd, y_pd
):
    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=sk_est_log,
        param_grid=param_grid_sk_log,
        scoring=['accuracy', 'balanced_accuracy'],
        refit=lambda x: 0
    )
    return __.fit(X_pd, y_pd)

# END gstcv log est two scorers, various refits


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


# END pipeline esimators ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def param_grid_pipe_sk_log():
    return {
        'sk_StandardScaler__with_mean': [True, False],
        'sk_logistic__C': [1e-4, 1e-5]
    }

# END pipe param grids ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# gscv log pipe one scorer, various refits
@pytest.fixture(scope='session')
def sk_GSCV_pipe_log_one_scorer_prefit(
    sk_gscv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
    sk_gscv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
    sk_gscv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
    sk_gscv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
    sk_gstcv_init_params, sk_est_log, sk_standard_scaler,
        param_grid_pipe_sk_log
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
    sk_gstcv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np(
    sk_gstcv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
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
def sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np(
    sk_gstcv_init_params, sk_est_log, sk_standard_scaler,
    param_grid_pipe_sk_log, X_np, y_np
):

    pipe = Pipeline(
        steps=[
            ('sk_StandardScaler', sk_standard_scaler),
            ('sk_logistic', sk_est_log)
        ]
    )

    __ = sk_GSTCV(**sk_gstcv_init_params)
    __.set_params(
        estimator=pipe,
        param_grid=param_grid_pipe_sk_log,
        refit=lambda x: 0
    )
    return __.fit(X_np, y_np)

# END gstcv log pipe one scorer, various refits


# END PIPELINES - ONE SCORER ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *







