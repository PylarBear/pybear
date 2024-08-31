# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
import sys, os
import string
from pathlib import Path


from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from xgboost import XGBClassifier as sk_XGBClassifier

from distributed import Client
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
from dask_ml.datasets import make_classification as dask_make_classification
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from xgboost.dask import DaskXGBClassifier as dask_XGBClassifier
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask

from getattr_handling.getattr_handling import (
    access_attrs,
    access_methods
)


# THIS MODULE CAPTURES THE OUTPUT (VALUE OR EXC INFO) FOR getattr CALLS
# TO ATTRS AND METHODS FOR SK GSCV, DASK GSCV, GSTCV, GSTCVDask
# BEFORE AND AFTER fit()
# THIS TAKES ABOUT 19 MINUTES TO RUN






dump_to_file = True

run_dtype = 'dataframe'  # array / dataframe
_scoring = ['accuracy', 'balanced_accuracy']   #'balanced_accuracy' #


if not run_dtype in ['array', 'dataframe']:
    raise Exception(f'run_dtype must be "array" or "dataframe"')

ROUND = ['post_init', 'post_fit']
REFIT = [False, 'balanced_accuracy', 'refit_fxn']
TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

COMBINATIONS = [f'{c}_refit_{b}_{a}' for a in ROUND for b in REFIT for c in TYPES]

# THIS ORDER MUST MATCH THE FILL ORDER IN access_attrs()
ATTR_NAMES = [
    'cv_results_', 'best_estimator_', 'best_score_', 'best_params_',
    'best_index_', 'scorer_', 'n_splits_', 'refit_time_', 'multimetric_',
    'classes_', 'n_features_in_', 'feature_names_in_', 'best_threshold_'
]

# THIS ORDER MUST MATCH THE FILL ORDER IN access_methods()
METHOD_NAMES = [
    'decision_function', 'get_metadata_routing', 'get_params',
    'inverse_transform', 'predict', 'predict_log_proba', 'predict_proba',
    'score', 'score_samples', 'set_params', 'transform', 'visualize'
]

ATTR_ARRAY_DICT = {_mix: np.empty((0,), dtype=object) for _mix in COMBINATIONS}
METHOD_ARRAY_DICT = {_mix: np.empty((0,), dtype=object) for _mix in COMBINATIONS}

_rows, _cols = 100, 10



da_X, da_y = dask_make_classification(
    n_samples=_rows,
    n_features=_cols,
    n_informative=_cols,
    n_redundant=0,
    chunks=(_rows // 10)
)

if run_dtype == 'dataframe':
    # DONT DO from_array, USE from_pandas TO PRESERVE COLUMN NAMES
    da_X = ddf.from_dask_array(
        da_X,
        columns=list(string.ascii_lowercase.replace('y', '')[:_cols])
    )
    da_y = ddf.from_dask_array(da_y, columns=['y'])  # .squeeze()


da_X_train, da_X_test, da_y_train, da_y_test = \
    dask_train_test_split(da_X, da_y, test_size=0.2)

try:
    da_y_train = da_y_train.to_dask_array(lengths=True).ravel()
    da_y_test = da_y_test.to_dask_array(lengths=True).ravel()
except:
    pass



sk_X_train = da_X_train.compute()
sk_X_test = da_X_test.compute()
sk_y_train = da_y_train.compute()
sk_y_test = da_y_test.compute()



# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO BUILD GRIDSEARCH INSTANCES AND ACCESS OUTPUT ** ** ** ** ** ** **

class OptimalParamsEst:

    """
    This class takes on any of the 4 gridsearch modules.
    During the trials, this class's optimal_params_est attr holds the
    gridsearch for the trial as set by the _gscv_type arg. The attrs
    and methods for the gridsearch (as contained in optimal_params_est)
    are exposed by the @propertys



    """

    def __init__(self, _gscv_type: str, _refit: str):

        __scoring = _scoring

        self.param_grid = {'C': np.logspace(-5,-2,4)} #{'max_depth': [3,4,5]}  #

        if _gscv_type == 'sklearn':
            self.optimal_params_est = sklearn_GridSearchCV(
                # estimator=sk_XGBClassifier(
                #     n_estimators=50,
                #     max_depth=5,
                #     tree_method='hist'
                # ),
                estimator=sk_LogisticRegression(
                    C=1e-3,
                    solver='lbfgs',
                    tol=1e-6,
                    fit_intercept=False
                ),
                param_grid=self.param_grid,
                cv=5,
                error_score='raise',
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'dask':
            self.optimal_params_est = dask_GridSearchCV(
                # estimator=sk_XGBClassifier(
                #     n_estimators=50,
                #     max_depth=5,
                #     tree_method='hist'
                # ),
                estimator=sk_LogisticRegression(
                    C=1e-3,
                    solver='lbfgs',
                    tol=1e-6,
                    fit_intercept=False
                ),
                param_grid=self.param_grid,
                cv=5,
                error_score='raise',
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'gstcv_sklearn':
            self.optimal_params_est = GSTCV(
                # estimator=sk_XGBClassifier(
                #     n_estimators=50,
                #     max_depth=5,
                #     tree_method='hist'
                # ),
                estimator=sk_LogisticRegression(
                    C=1e-3,
                    solver='lbfgs',
                    tol=1e-6,
                    fit_intercept=False
                ),
                param_grid=self.param_grid,
                thresholds=[0.5],
                cv=5,
                error_score='raise',
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'gstcv_dask':
            self.optimal_params_est = GSTCVDask(
                # estimator=sk_XGBClassifier(
                #     n_estimators=50,
                #     max_depth=5,
                #     tree_method='hist'
                # ),
                estimator=sk_LogisticRegression(
                    C=1e-3,
                    solver='lbfgs',
                    tol=1e-6,
                    fit_intercept=False
                ),
                param_grid=self.param_grid,
                thresholds=[0.5],
                cv=5,
                error_score='raise',
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

    @property
    def best_threshold_(self):
        return self.optimal_params_est.best_threshold_

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
        return self.optimal_params_est.visualize(filename=filename, format=format)

    def fit(self, X, y):
        return self.optimal_params_est.fit(X, y)


# END FUNCTION TO BUILD GRIDSEARCH INSTANCES AND ACCESS OUTPUT ** ** ** ** ** *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **





# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** ** ** *

def refit_fxn(cv_results_):
    DF = pd.DataFrame(cv_results_)
    try:
        __ = DF.index[DF['rank_test_balanced_accuracy'] == 1][0]
    except:
        __ = DF.index[DF['rank_test_score'] == 1][0]

    return int(__)


def handler_for_getattr(
    access_attrs_methods_or_both: str,
    ROUND: list[str],
    REFIT: list[str],
    TYPES: list[str],
    sk_X_train,
    sk_y_train,
    da_X_train,
    da_y_train,
    sk_X_test,
    sk_y_test,
    da_X_test,
    da_y_test,
    dump_to_file,
    ATTR_ARRAY_DICT,
    METHOD_ARRAY_DICT
):

    for _round in ROUND:

        for _refit in REFIT:
            refit_words = _refit
            _refit = refit_fxn if _refit == 'refit_fxn' else _refit
            for _gscv_type in TYPES:

                itr = f'{_gscv_type}_refit_{refit_words}_{_round}'
                # print(itr)

                if 'sklearn' in _gscv_type:
                    for _ in (sk_X_train, sk_y_train, sk_X_test, sk_y_test):
                        assert isinstance(_, (np.ndarray, pd.core.frame.DataFrame))

                elif 'dask' in _gscv_type:
                    for _ in (da_X_train, da_y_train, da_X_test, da_y_test):
                        assert isinstance(
                            _,
                            (da.core.Array, ddf.core.DataFrame, ddf2.DataFrame)
                        )


                test_cls = OptimalParamsEst(_gscv_type, _refit)

                if _round == 'post_fit':

                    try:
                        if 'sklearn' in _gscv_type:
                            test_cls.fit(sk_X_train, sk_y_train)
                        elif 'dask' in _gscv_type:
                            test_cls.fit(da_X_train, da_y_train)
                    except:
                        e = sys.exc_info()[1]
                        ATTR_ARRAY_DICT[itr] = [f'{e!r}' for _ in ATTR_NAMES]
                        METHOD_ARRAY_DICT[itr] = [f'{e!r}' for _ in METHOD_NAMES]

                        continue

                attr_args = (_round, _gscv_type, refit_words, dump_to_file)

                if access_attrs_methods_or_both in ['attrs', 'both']:
                    ATTR_ARRAY_DICT = access_attrs(
                        test_cls,
                        *attr_args,
                        ATTR_ARRAY_DICT
                    )

                if access_attrs_methods_or_both in ['methods', 'both']:
                    if 'sklearn' in _gscv_type:
                        METHOD_ARRAY_DICT = access_methods(
                            test_cls,
                            sk_X_test,
                            sk_y_test,
                            *attr_args,
                            METHOD_ARRAY_DICT
                        )
                    elif 'dask' in _gscv_type:
                        METHOD_ARRAY_DICT = access_methods(
                            test_cls,
                            da_X_test,
                            da_y_test,
                            *attr_args,
                            METHOD_ARRAY_DICT
                        )

                del test_cls, attr_args

    return ATTR_ARRAY_DICT, METHOD_ARRAY_DICT






if __name__ == '__main__':


    # WHEN PRINTING TO SCREEN, KEEP access_attrs & access_methods SEPARATE SO THAT
    # attr/method OUTPUT ISNT INTERMINGLED ** ** ** ** ** **

    with Client(n_workers=None, threads_per_worker=1):

        args = (ROUND, REFIT, TYPES, sk_X_train, sk_y_train, da_X_train,
                da_y_train, sk_X_test, sk_y_test, da_X_test, da_y_test,
                dump_to_file, ATTR_ARRAY_DICT, METHOD_ARRAY_DICT)

        if not dump_to_file:
            ATTR_ARRAY_DICT, METHOD_ARRAY_DICT = handler_for_getattr('attrs', *args)

            ATTR_ARRAY_DICT, METHOD_ARRAY_DICT = handler_for_getattr('methods', *args)
        # END WHEN PRINTING TO SCREEN ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        elif dump_to_file:
            ATTR_ARRAY_DICT, METHOD_ARRAY_DICT = handler_for_getattr('both', *args)

        del args

    ###########################################################################
    # WRITE ATTR RESULTS TO FILE ##############################################
    SINGLE_DF = pd.DataFrame(
        index=ATTR_NAMES,
        columns=list(ATTR_ARRAY_DICT.keys()),
        dtype='<U100'
    ).fillna('-')

    for _key, DATA_COLUMN in ATTR_ARRAY_DICT.items():
        SINGLE_DF.loc[:, _key] = DATA_COLUMN

    SINGLE_DF = SINGLE_DF.T

    desktop_path = Path.home() / "Desktop"
    filename = rf'gscv_attr_comparison_dump.csv'
    attr_path = desktop_path / filename

    SINGLE_DF.to_csv(attr_path, index=True)
    # END WRITE ATTR RESULTS TO FILE ##########################################
    ###########################################################################

    ###########################################################################
    # WRITE METHOD RESULTS TO FILE ############################################
    SINGLE_DF = pd.DataFrame(
        index=METHOD_NAMES,
        columns=list(METHOD_ARRAY_DICT.keys()),
        dtype='<U100'
    ).fillna('-')

    for _key, DATA_COLUMN in METHOD_ARRAY_DICT.items():
        SINGLE_DF.loc[:, _key] = DATA_COLUMN

    SINGLE_DF = SINGLE_DF.T

    desktop_path = Path.home() / "Desktop"
    filename = rf'gscv_method_comparison_dump.csv'
    method_path = desktop_path / filename

    SINGLE_DF.to_csv(method_path, index=True)
    # END WRITE METHOD RESULTS TO FILE ########################################
    ###########################################################################

# END DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **























