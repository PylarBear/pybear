# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
import sys, os, base64, io, time
import scipy.sparse as ss
import functools
import string

from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LogisticRegression as sklearn_Logistic

import dask.array as da
import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.datasets import make_classification as dask_make_classification
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_Logistic

from model_selection.GSTCV._GSTCV import GridSearchThresholdCV as GSTCV

dump_to_file = True

run_dtype = 'dataframe'  # array / dataframe

if not run_dtype in ['array', 'dataframe']:
    raise Exception(f'run_dtype must be "array" or "dataframe"')

ROUND = ['post_init', 'post_fit']  # ['post_fit']
REFIT = [False, 'balanced_accuracy', 'refit_fxn']  # ['balanced_accuracy']
TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

COMBINATIONS = [f'{c}_refit_{b}_{a}' for a in ROUND for b in REFIT for c in
                TYPES]

# THIS ORDER MUST MATCH THE FILL ORDER IN access_attrs()
ATTR_NAMES = [
    'cv_results_', 'best_estimator_', 'best_score_', 'best_params_',
    'best_index_', 'scorer_',
    'n_splits_', 'refit_time_', 'multimetric_', 'classes_', 'n_features_in_',
    'feature_names_in_'
]

# THIS ORDER MUST MATCH THE FILL ORDER IN access_methods()
METHOD_NAMES = ['decision_function', 'get_metadata_routing', 'get_params',
                'inverse_transform', 'predict',
                'predict_log_proba', 'predict_proba', 'score', 'score_samples',
                'set_params', 'transform',
                'visualize'
                ]

ATTR_ARRAY_DICT = {_mix: np.empty((0,), dtype=object) for _mix in COMBINATIONS}
METHOD_ARRAY_DICT = {_mix: np.empty((0,), dtype=object) for _mix in
                     COMBINATIONS}
CV_RESULTS_DICT = {_mix: {} for _mix in COMBINATIONS}

_rows, _cols = 100, 10

sk_X, sk_y = sklearn_make_classification(n_samples=_rows, n_features=_cols,
                                         n_informative=_cols, n_redundant=0)

if run_dtype == 'dataframe':
    sk_X = pd.DataFrame(data=sk_X, columns=list(
        string.ascii_lowercase.replace('y', '')[:_cols]))
    sk_y = pd.DataFrame(data=sk_y, columns=['y'])  # .squeeze()

sk_X1, sk_X_test, sk_y1, sk_y_test = sklearn_train_test_split(sk_X, sk_y,
                                                              test_size=0.2)
sk_X_train, sk_X_val, sk_y_train, sk_y_val = sklearn_train_test_split(sk_X1,
                                                                      sk_y1,
                                                                      test_size=0.25)

if run_dtype == 'dataframe':
    da_X = ddf.from_pandas(sk_X,
                           npartitions=_rows // 10)  # DONT DO from_array, USE THIS TO PRESERVE COLUMN NAMES
    da_y = ddf.from_pandas(sk_y, npartitions=_rows // 10)  # .squeeze()
else:
    da_X, da_y = da.array(sk_X).rechunk((_rows // 10, _cols)), da.array(
        sk_y).rechunk((_rows // 10, _cols))

da_X1, da_X_test, da_y1, da_y_test = dask_train_test_split(da_X, da_y,
                                                           test_size=0.2)
da_X_train, da_X_val, da_y_train, da_y_val = dask_train_test_split(da_X1,
                                                                   da_y1,
                                                                   test_size=0.25)

del sk_X1, sk_y1
del da_X1, da_y1


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO BUILD GRIDSEARCH INSTANCES AND ACCESS OUTPUT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

class OptimalParamsEst:
    def __init__(self, _gscv_type: str, _refit: str):

        #  BEAR 24_02_26_07_58_00 STILL HAVE TO RUN 1 METRIC
        __scoring = ['accuracy',
                     'balanced_accuracy']  # 'balanced_accuracy'   #

        self.param_grid = {'C': np.logspace(-2, 2, 5)}

        if _gscv_type == 'sklearn':
            self.optimal_params_est = sklearn_GridSearchCV(
                estimator=sklearn_Logistic(max_iter=10_000, tol=1e-6,
                                           solver='lbfgs'),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'dask':
            self.optimal_params_est = dask_GridSearchCV(
                estimator=dask_Logistic(fit_intercept=False, max_iter=10_000,
                                        tol=1e-6, solver='lbfgs'),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'gstcv_sklearn':
            self.optimal_params_est = GSTCV(
                estimator=sklearn_Logistic(max_iter=10_000, tol=1e-6,
                                           solver='lbfgs'),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'gstcv_dask':
            self.optimal_params_est = GSTCV(
                estimator=dask_Logistic(fit_intercept=False, max_iter=10_000,
                                        tol=1e-6, solver='lbfgs'),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit
            )

    @property
    def cv_results_(self):
        return self.optimal_params_est.cv_results_

    @property
    def best_estimator_(self):
        return self.optimal_params_est.best_estimator_

    @property
    def best_score_(self):
        return self.optimal_params_est.best_score_

    @property
    def best_params_(self):
        return self.optimal_params_est.best_params_

    @property
    def best_index_(self):
        return self.optimal_params_est.best_index_

    @property
    def scorer_(self):
        return self.optimal_params_est.scorer_

    @property
    def n_splits_(self):
        return self.optimal_params_est.n_splits_

    @property
    def refit_time_(self):
        return self.optimal_params_est.refit_time_

    @property
    def multimetric_(self):
        return self.optimal_params_est.multimetric_

    @property
    def classes_(self):
        return self.optimal_params_est.classes_

    @property
    def n_features_in_(self):
        return self.optimal_params_est.n_features_in_

    @property
    def feature_names_in_(self):
        return self.optimal_params_est.feature_names_in_

    def decision_function(self, X):
        return self.optimal_params_est.decision_function(X)

    def get_metadata_routing(self):  # --- sklearn only
        return self.optimal_params_est.get_metadata_routing()

    def get_params(self, deep: bool = True):
        return self.optimal_params_est.get_params(deep)

    def inverse_transform(self, Xt):
        return self.optimal_params_est.inverse_transform(Xt)

    def predict(self, X):
        return self.optimal_params_est.predict(X)

    def predict_log_proba(self, X):
        return self.optimal_params_est.predict_log_proba(X)

    def predict_proba(self, X):
        return self.optimal_params_est.predict_proba(X)

    def score(self, X, y=None, **params):
        return self.optimal_params_est.score(X, y, **params)

    def score_samples(self, X):  # --- sklearn only
        return self.optimal_params_est.score_samples(X)

    def set_params(self, **params):
        return self.optimal_params_est.set_params(**params)

    def transform(self, X):
        return self.optimal_params_est.transform(X)

    def visualize(self, filename=None, format=None):  # --- dask only
        return self.optimal_params_est.visualize(filename=None, format=None)

    def fit(self, X, y):
        return self.optimal_params_est.fit(X, y)


# END FUNCTION TO BUILD GRIDSEARCH INSTANCES AND ACCESS OUTPUT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO DISPLAY GRIDSEARCH ATTR OUTPUT B4 & AFTER CALLS TO fit() ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

def access_attrs(
        xgscv_instance,
        _round: str,
        _gscv_type: str,
        _refit: bool,
        dump_to_file: bool,
        ATTR_ARRAY_DICT: dict
):
    _exc = lambda: sys.exc_info()[1]

    def print_or_dump_try_handling(attr_output: any, attr_name: str,
                                   _gscv_type: str, _refit: bool,
                                   dump_to_file: bool,
                                   ATTR_ARRAY_DICT: dict) -> dict:
        if not dump_to_file:  # print to screen
            print(f"{attr_name} = {attr_output}")
        else:
            _key = f"{_gscv_type}_refit_{_refit}_{_round}"

            if _key not in ATTR_ARRAY_DICT:
                raise Exception(
                    f"attempting to write to ATTR_ARRAY_DICT key '{_key}' but it doesnt exist")

            ATTR_ARRAY_DICT[_key] = np.insert(ATTR_ARRAY_DICT[_key],
                                              len(ATTR_ARRAY_DICT[_key]),
                                              str(attr_output), axis=0)

            del _key

        return ATTR_ARRAY_DICT

    def print_or_dump_except_handling(exception_info: str, attr_name: str,
                                      _gscv_type: str, _refit: bool,
                                      dump_to_file: bool,
                                      ATTR_ARRAY_DICT: dict) -> dict:
        if not dump_to_file:  # print to screen
            print(f"{attr_name}: {exception_info}")
        else:

            _key = f"{_gscv_type}_refit_{_refit}_{_round}"

            if _key not in ATTR_ARRAY_DICT:
                raise Exception(
                    f"attempting to write to ATTR_ARRAY_DICT key '{_key}' but it doesnt exist")

            ATTR_ARRAY_DICT[_key] = np.insert(ATTR_ARRAY_DICT[_key],
                                              len(ATTR_ARRAY_DICT[_key]),
                                              str(exception_info), axis=0)

            del _key

        return ATTR_ARRAY_DICT

    if not dump_to_file: print(
        f"\nStart attr round {_round} {_gscv_type} refit {_refit} " + f"** " * 20 + f"\n")

    try:
        xgscv_instance.cv_results_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.cv_results_["params"][0],
            f'cv_results_["params"][0]', _gscv_type, _refit, dump_to_file,
            ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"cv_results_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.best_estimator_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.best_estimator_, f"best_estimator_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                        f"best_estimator_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.best_score_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.best_score_, f"best_score_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"best_score_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.best_params_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.best_params_, f"best_params_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                        f"best_params_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.best_index_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.best_index_, f"best_index_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"best_index_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.scorer_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(xgscv_instance.scorer_,
                                                     f"scorer_",
                                                     _gscv_type, _refit,
                                                     dump_to_file,
                                                     ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"scorer_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.n_splits_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(xgscv_instance.n_splits_,
                                                     f"n_splits_",
                                                     _gscv_type, _refit,
                                                     dump_to_file,
                                                     ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"n_splits_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.refit_time_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.refit_time_, f"refit_time_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"refit_time_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.multimetric_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.multimetric_, f"multimetric_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                        f"multimetric_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.classes_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(xgscv_instance.classes_,
                                                     f"classes_",
                                                     _gscv_type, _refit,
                                                     dump_to_file,
                                                     ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"classes_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.n_features_in_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.n_features_in_, f"n_features_in_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                        f"n_features_in_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    try:
        xgscv_instance.feature_names_in_
        ATTR_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.feature_names_in_, f"feature_names_in_",
            _gscv_type, _refit, dump_to_file, ATTR_ARRAY_DICT)
    except:
        ATTR_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                        f"feature_names_in_",
                                                        _gscv_type, _refit,
                                                        dump_to_file,
                                                        ATTR_ARRAY_DICT)

    del _exc, print_or_dump_try_handling, print_or_dump_except_handling

    if not dump_to_file: print(
        f"\nEnd attr round {_round} {_gscv_type} refit {_refit} " + f"** " * 20,
        f"\n")

    return ATTR_ARRAY_DICT


# END FUNCTION TO DISPLAY GRIDSEARCH ATTR OUTPUT B4 & AFTER CALLS TO fit() ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO DISPLAY GRIDSEARCH METHOD OUTPUT B4 & AFTER CALLS TO fit() ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

# THESE METHODS NAMES ARE VERIFIED & UP-TO-DATE AS OF 24_02_19_09_26_00

# self.decision_function(X)
# self.get_metadata_routing() --- sklearn only
# self.get_params(deep:bool=True)
# self.inverse_transform(Xt)
# self.predict(X)
# self.predict_log_proba(X)
# self.predict_proba(X)
# self.score(X, y=None, **params)
# self.score_samples(X) --- sklearn only
# self.set_params(**params)
# self.transform(X)
# self.visualize(filename=None, format=None) --- dask only


def access_methods(
        xgscv_instance,
        X,
        y,
        _round: str,
        _gscv_type: str,
        _refit: str,
        dump_to_file: bool,
        METHOD_ARRAY_DICT: dict,
        **score_params
):
    _exc = lambda: sys.exc_info()[1]

    def print_or_dump_try_handling(method_output: any, method_name: str,
                                   _gscv_type: str, _refit: bool,
                                   dump_to_file: bool,
                                   METHOD_ARRAY_DICT: dict) -> dict:

        if not dump_to_file:  # print to screen
            print(f"{method_name} = {method_output}")
        else:
            _key = f"{_gscv_type}_refit_{_refit}_{_round}"

            if _key not in METHOD_ARRAY_DICT:
                raise Exception(
                    f"attempting to write to METHOD_ARRAY_DICT key '{_key}' but it doesnt exist")

            METHOD_ARRAY_DICT[_key] = np.insert(METHOD_ARRAY_DICT[_key],
                                                len(METHOD_ARRAY_DICT[_key]),
                                                str(method_output), axis=0)

            del _key

        return METHOD_ARRAY_DICT

    def print_or_dump_except_handling(exception_info: str, method_name: str,
                                      _gscv_type: str, _refit: bool,
                                      dump_to_file: bool,
                                      METHOD_ARRAY_DICT: dict) -> dict:
        if not dump_to_file:  # print to screen
            if 'visualize' in method_name.lower() and 'sklearn' in _gscv_type.lower() or \
                    'get_metadata_routing' in method_name.lower() and 'dask' in _gscv_type.lower() or \
                    'score_samples' in method_name.lower() and 'dask' in _gscv_type.lower():
                print(f"{method_name} = {exception_info}")
            else:
                print(f"{method_name}: {exception_info}")
        else:

            _key = f"{_gscv_type}_refit_{_refit}_{_round}"

            if _key not in METHOD_ARRAY_DICT:
                raise Exception(
                    f"attempting to write to METHOD_ARRAY_DICT key '{_key}' but it doesnt exist")

            if 'visualize' in method_name.lower() and 'sklearn' in _gscv_type.lower() or \
                    'get_metadata_routing' in method_name.lower() and 'dask' in _gscv_type.lower() or \
                    'score_samples' in method_name.lower() and 'dask' in _gscv_type.lower():
                METHOD_ARRAY_DICT[_key] = np.insert(METHOD_ARRAY_DICT[_key],
                                                    len(METHOD_ARRAY_DICT[
                                                            _key]),
                                                    str(exception_info),
                                                    axis=0)
            else:
                METHOD_ARRAY_DICT[_key] = np.insert(METHOD_ARRAY_DICT[_key],
                                                    len(METHOD_ARRAY_DICT[
                                                            _key]),
                                                    str(exception_info),
                                                    axis=0)

            del _key

        return METHOD_ARRAY_DICT

    if not dump_to_file: print(
        f"\nStart method round {_round} {_gscv_type} refit {_refit} " + f"** " * 20 + f"\n")

    try:
        xgscv_instance.decision_function(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.decision_function(X), f"decision_function",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"decision_function",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.get_metadata_routing()
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.get_metadata_routing(), f"get_metadata_routing",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"get_metadata_routing",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.get_params(deep=True)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.get_params(deep=True), f"get_params",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"get_params",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.inverse_transform(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.inverse_transform(X), f"inverse_transform",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"inverse_transform",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.predict(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.predict(X), f"predict",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"predict",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.predict_log_proba(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.predict_log_proba(X), f"predict_log_proba",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"predict_log_proba",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.predict_proba(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.predict_proba(X), f"predict_proba",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"predict_proba",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.score(X, y, **score_params)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.score(X, y, **score_params), f"score",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"score",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.score_samples(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.score_samples(X), f"score_samples",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"score_samples",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.set_params(estimator__C=100)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.set_params(estimator__C=100), f"set_params",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(),
                                                          f"set_params",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.transform(X)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.transform(X), f"transform",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"transform",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    try:
        xgscv_instance.visualize(filename=None, format=None)
        METHOD_ARRAY_DICT = print_or_dump_try_handling(
            xgscv_instance.visualize(filename=None, format=None), f"visualize",
            _gscv_type, _refit, dump_to_file, METHOD_ARRAY_DICT)
    except:
        METHOD_ARRAY_DICT = print_or_dump_except_handling(_exc(), f"visualize",
                                                          _gscv_type, _refit,
                                                          dump_to_file,
                                                          METHOD_ARRAY_DICT)

    if not dump_to_file: print(
        f"End method round {_round} {_gscv_type} refit {_refit} " + f"** " * 20 + f"\n")

    return METHOD_ARRAY_DICT


# END FUNCTION TO DISPLAY GRIDSEARCH METHOD OUTPUT B4 & AFTER CALLS TO fit() ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

def refit_fxn(cv_results_):
    # DF = pd.DataFrame(cv_results_)
    # try: print(DF['rank_test_score'])
    # except: pass
    # try: print(DF['rank_test_balanced_accuracy'])
    # except: pass
    return 0
    # return DF.index[DF['rank_test_balanced_accuracy']==1][0]


def generic_handler_for_print_or_dump(ROUND, REFIT, TYPES, sk_X_train,
                                      sk_y_train, da_X_train, da_y_train,
                                      sk_X_test, sk_y_test, da_X_test,
                                      da_y_test,
                                      dump_to_file,
                                      access_attrs_methods_or_both,
                                      ATTR_ARRAY_DICT, METHOD_ARRAY_DICT,
                                      CV_RESULTS_DICT):
    for _round in ROUND:

        for _refit in REFIT:
            refit_words = _refit
            _refit = refit_fxn if _refit == 'refit_fxn' else _refit
            for _gscv_type in TYPES:

                print(f"{_gscv_type}_refit_{refit_words}_{_round}")

                if 'dask' in _gscv_type:
                    _msg = ""
                    if _gscv_type == 'dask' and _round == 'post_fit' and refit_words == 'refit_fxn':
                        _msg = 'dask cannot take callable for refit'

                    if _msg != "":
                        if not dump_to_file:
                            print(
                                f'{_gscv_type}_refit_{refit_words}_{_round}: {_msg}')
                        else:
                            CV_RESULTS_DICT[
                                f"{_gscv_type}_refit_{refit_words}_{_round}"] = {
                                f'{_msg}': [np.nan]}
                            ATTR_ARRAY_DICT[
                                f"{_gscv_type}_refit_{refit_words}_{_round}"] = [
                                f'{_msg}' for _ in ATTR_NAMES]
                            METHOD_ARRAY_DICT[
                                f"{_gscv_type}_refit_{refit_words}_{_round}"] = [
                                f'{_msg}' for _ in METHOD_NAMES]

                        continue

                test_cls = OptimalParamsEst(_gscv_type, _refit)

                if _round == 'post_fit':

                    try:
                        if 'sklearn' in _gscv_type:
                            test_cls.fit(sk_X_train, sk_y_train)
                        elif 'dask' in _gscv_type:
                            test_cls.fit(da_X_train, da_y_train)
                    except:
                        CV_RESULTS_DICT[
                            f"{_gscv_type}_refit_{refit_words}_{_round}"] = {
                            f'{sys.exc_info()[1]}': [np.nan]}
                        ATTR_ARRAY_DICT[
                            f"{_gscv_type}_refit_{refit_words}_{_round}"] = [
                            f'{sys.exc_info()[1]}' for _ in ATTR_NAMES]
                        METHOD_ARRAY_DICT[
                            f"{_gscv_type}_refit_{refit_words}_{_round}"] = [
                            f'{sys.exc_info()[1]}' for _ in METHOD_NAMES]

                        continue

                if dump_to_file:
                    try:
                        CV_RESULTS_DICT[
                            f"{_gscv_type}_refit_{refit_words}_{_round}"] = test_cls.cv_results_
                    except:
                        CV_RESULTS_DICT[
                            f"{_gscv_type}_refit_{refit_words}_{_round}"] = {
                            'cv_results_ not available': [np.nan]}

                if access_attrs_methods_or_both in ['attrs', 'both']:
                    ATTR_ARRAY_DICT = access_attrs(test_cls, _round,
                                                   _gscv_type, refit_words,
                                                   dump_to_file,
                                                   ATTR_ARRAY_DICT)

                if access_attrs_methods_or_both in ['methods', 'both']:
                    if 'sklearn' in _gscv_type:
                        METHOD_ARRAY_DICT = access_methods(test_cls, sk_X_test,
                                                           sk_y_test, _round,
                                                           _gscv_type,
                                                           refit_words,
                                                           dump_to_file,
                                                           METHOD_ARRAY_DICT)
                    elif 'dask' in _gscv_type:
                        METHOD_ARRAY_DICT = access_methods(test_cls, da_X_test,
                                                           da_y_test, _round,
                                                           _gscv_type,
                                                           refit_words,
                                                           dump_to_file,
                                                           METHOD_ARRAY_DICT)

                del test_cls

    return ATTR_ARRAY_DICT, METHOD_ARRAY_DICT, CV_RESULTS_DICT


# WHEN PRINTING TO SCREEN, KEEP access_attrs & access_methods SEPARATE SO THAT attr/method OUTPUT ISNT INTERMINGLED ** ** ** ** ** **
if not dump_to_file:
    ATTR_ARRAY_DICT, METHOD_ARRAY_DICT, CV_RESULTS_DICT = \
        generic_handler_for_print_or_dump(ROUND, REFIT, TYPES, sk_X_train,
                                          sk_y_train, da_X_train, da_y_train,
                                          sk_X_test, sk_y_test, da_X_test,
                                          da_y_test,
                                          dump_to_file, 'attrs',
                                          ATTR_ARRAY_DICT, METHOD_ARRAY_DICT,
                                          CV_RESULTS_DICT)

    ATTR_ARRAY_DICT, METHOD_ARRAY_DICT, CV_RESULTS_DICT = \
        generic_handler_for_print_or_dump(ROUND, REFIT, TYPES, sk_X_train,
                                          sk_y_train, da_X_train, da_y_train,
                                          sk_X_test, sk_y_test, da_X_test,
                                          da_y_test,
                                          dump_to_file, 'methods',
                                          ATTR_ARRAY_DICT, METHOD_ARRAY_DICT,
                                          CV_RESULTS_DICT)
# END WHEN PRINTING TO SCREEN ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

elif dump_to_file:
    ATTR_ARRAY_DICT, METHOD_ARRAY_DICT, CV_RESULTS_DICT = \
        generic_handler_for_print_or_dump(ROUND, REFIT, TYPES, sk_X_train,
                                          sk_y_train, da_X_train, da_y_train,
                                          sk_X_test, sk_y_test, da_X_test,
                                          da_y_test,
                                          dump_to_file, 'both',
                                          ATTR_ARRAY_DICT, METHOD_ARRAY_DICT,
                                          CV_RESULTS_DICT)

    ############################################################################################################################
    # WRITE ATTR RESULTS TO FILE ###############################################################################################
    # INDIVIDUAL FILES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    #
    #       ** ** ** ** ** ** ** ** NOTICE THAT 'ATTR_TESTS' FOLDER MUST EXIST ON DESKTOP ** ** ** ** ** ** ** ** ** ** ** **
    # if os.name=='posix': attr_path = rf'/home/bear/Desktop/ATTR_TESTS/{sheet_name}.ods'
    # elif os.name=='nt': attr_path = rf'c:\users\bill\desktop\ATTR_TESTS\{sheet_name}.csv'
    #
    # for sheet_name, results in ATTR_ARRAY_DICT.items():
    #     DF = pd.DataFrame(data=results, index=ATTR_NAMES, columns=['OUTPUT'])
    #     DF.to_csv(attr_path, index=True)
    # END INDIVIDUAL FILES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ExcelWriter ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # if os.name =='posix':
    #     attr_path = rf'/home/bear/Desktop/gscv_attr_comparison_dump.ods'
    #     attr_engine=None
    # elif os.name=='nt':
    #     attr_path = rf'c:\users\bill\desktop\gscv_attr_comparison_dump.csv'
    #     attr_engine='openpyxl'

    # with pd.ExcelWriter(attr_path, engine=attr_engine, mode='w') as writer:
    #     for _sheet_name, results in ATTR_ARRAY_DICT.items():
    #         DF = pd.DataFrame(data=results, index=ATTR_NAMES, columns=['OUTPUT'])
    #         DF.to_excel(writer, sheet_name=_sheet_name)
    # END ExcelWriter ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ONE SHEET ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    SINGLE_DF = pd.DataFrame(index=ATTR_NAMES,
                             columns=list(ATTR_ARRAY_DICT.keys()),
                             dtype='<U100').fillna('-')
    for _key, DATA_COLUMN in ATTR_ARRAY_DICT.items():
        print(f'BEAR TEST DATA_COLUMN = ')
        print(DATA_COLUMN)
        print(f'BEAR TEST SINGLE_DF.index = ')
        print(SINGLE_DF.index)
        SINGLE_DF.loc[:, _key] = DATA_COLUMN

    if os.name == 'posix':
        attr_path = rf'/home/bear/Desktop/gscv_attr_comparison_dump.ods'
    elif os.name == 'nt':
        attr_path = rf'c:\users\bill\desktop\gscv_attr_comparison_dump.csv'

    SINGLE_DF.to_csv(attr_path, index=True)
    # END ONE SHEET ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # END WRITE ATTR RESULTS TO FILE #############################################################################################
    ##############################################################################################################################

    ##############################################################################################################################
    # WRITE METHOD RESULTS TO FILE ###############################################################################################
    # INDIVIDUAL FILES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    #
    #       ** ** ** ** ** ** ** ** NOTICE THAT 'METHOD_TESTS' FOLDER MUST EXIST ON DESKTOP ** ** ** ** ** ** ** **
    # for sheet_name, results in METHOD_ARRAY_DICT.items():
    #     DF = pd.DataFrame(data=results, index=METHOD_NAMES, columns=['OUTPUT'])
    #     if os.name=='posix': DF.to_csv(rf'/home/bear/Desktop/METHOD_TESTS/{sheet_name}.ods', index=True)
    #     elif os.name=='nt': DF.to_csv(rf'c:\users\bill\desktop\METHOD_TESTS\{sheet_name}.csv', index=True)
    # END INDIVIDUAL FILES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ONE SHEET ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    SINGLE_DF = pd.DataFrame(index=METHOD_NAMES,
                             columns=list(METHOD_ARRAY_DICT.keys()),
                             dtype='<U100').fillna('-')
    for _key, DATA_COLUMN in METHOD_ARRAY_DICT.items():
        SINGLE_DF.loc[:, _key] = DATA_COLUMN

    if os.name == 'posix':
        method_path = rf'/home/bear/Desktop/gscv_method_comparison_dump.ods'
    elif os.name == 'nt':
        method_path = rf'c:\users\bill\desktop\gscv_method_comparison_dump.csv'

    SINGLE_DF.to_csv(method_path, index=True)
    # END ONE SHEET ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # END WRITE METHOD RESULTS TO FILE ###########################################################################################
    ##############################################################################################################################

    ##############################################################################################################################
    # WRITE cv_results TO FILE ###################################################################################################

    # ExcelWriter ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    if os.name == 'posix':
        csv_results_path = rf'/home/bear/Desktop/gscv_cv_results_comparison_dump.ods'
        csv_results_engine = None
    elif os.name == 'nt':
        csv_results_path = rf'c:\users\bill\desktop\gscv_cv_results_comparison_dump.csv'
        csv_results_engine = 'openpyxl'

    with pd.ExcelWriter(csv_results_path, engine=csv_results_engine,
                        mode='w') as writer:
        for _sheet_name, results in CV_RESULTS_DICT.items():
            DF = pd.DataFrame(results)
            DF.to_excel(writer, sheet_name=_sheet_name)
    # END ExcelWriter ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # END WRITE cv_results TO FILE ################################################################################################
    ###############################################################################################################################

# END DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **









