# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
import sys, os
import string

from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
# pizza
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from xgboost import XGBClassifier as sk_XGBClassifier

from distributed import Client
import dask.array as da
import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
# pizza
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







dump_to_file = True

run_dtype = 'array'  # array / dataframe

if not run_dtype in ['array', 'dataframe']:
    raise Exception(f'run_dtype must be "array" or "dataframe"')

ROUND = ['post_init', 'post_fit']  # ['post_fit']
REFIT = [False, 'balanced_accuracy', 'refit_fxn']  # ['balanced_accuracy']
TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

COMBINATIONS = [f'{c}_refit_{b}_{a}' for a in ROUND for b in REFIT for c in TYPES]

# THIS ORDER MUST MATCH THE FILL ORDER IN access_attrs()
ATTR_NAMES = [
    'cv_results_', 'best_estimator_', 'best_score_', 'best_params_',
    'best_index_', 'scorer_', 'n_splits_', 'refit_time_', 'multimetric_',
    'classes_', 'n_features_in_', 'feature_names_in_'
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

sk_X, sk_y = sklearn_make_classification(
    n_samples=_rows,
    n_features=_cols,
    n_informative=_cols,
    n_redundant=0
)

if run_dtype == 'dataframe':
    sk_X = pd.DataFrame(
        data=sk_X,
        columns=list(string.ascii_lowercase.replace('y', '')[:_cols])
    )
    sk_y = pd.DataFrame(
        data=sk_y,
        columns=['y']
    )  # .squeeze()

sk_X1, sk_X_test, sk_y1, sk_y_test = \
    sklearn_train_test_split(sk_X, sk_y, test_size=0.2)
sk_X_train, sk_X_val, sk_y_train, sk_y_val = \
    sklearn_train_test_split(sk_X1, sk_y1, test_size=0.25)

if run_dtype == 'dataframe':
    # DONT DO from_array, USE from_pandas TO PRESERVE COLUMN NAMES
    da_X = ddf.from_pandas(sk_X, npartitions=_rows // 10)
    da_y = ddf.from_pandas(sk_y, npartitions=_rows // 10)  # .squeeze()
else:
    da_X = da.array(sk_X).rechunk((_rows // 10, _cols))
    da_y = da.array(sk_y).rechunk((_rows // 10, _cols))

da_X1, da_X_test, da_y1, da_y_test = \
    dask_train_test_split(da_X, da_y, test_size=0.2)
da_X_train, da_X_val, da_y_train, da_y_val = \
    dask_train_test_split(da_X1, da_y1, test_size=0.25)

del sk_X1, sk_y1
del da_X1, da_y1


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

        __scoring = ['accuracy', 'balanced_accuracy']

        self.param_grid = {'max_depth': [3,4,5]}  # {'C': np.logspace(-5,-2,4)}

        if _gscv_type == 'sklearn':
            self.optimal_params_est = sklearn_GridSearchCV(
                estimator=sk_XGBClassifier(n_estimators=50, tree_method='hist'),
                # estimator=sk_LogisticRegression(C=1e-3, solver='lbfgs', tol=1e-4),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'dask':
            self.optimal_params_est = dask_GridSearchCV(
                estimator=sk_XGBClassifier(n_estimators=50, tree_method='hist'),
                # estimator=dask_LogisticRegression(C=1e-3, solver='lbfgs', tol=1e-4),
                param_grid=self.param_grid,
                scoring=__scoring,
                refit=_refit,
                # scheduler=scheduler  # pizza
            )
        elif _gscv_type == 'gstcv_sklearn':
            self.optimal_params_est = GSTCV(
                estimator=sk_XGBClassifier(n_estimators=50, tree_method='hist'),
                # estimator=sk_LogisticRegression(C=1e-3, solver='lbfgs', tol=1e-4),
                param_grid=self.param_grid,
                thresholds=[0.5],
                scoring=__scoring,
                refit=_refit
            )
        elif _gscv_type == 'gstcv_dask':
            self.optimal_params_est = GSTCVDask(
                estimator=sk_XGBClassifier(n_estimators=50, tree_method='hist'),
                # estimator=dask_LogisticRegression(C=1e-3, solver='lbfgs', tol=1e-4),
                param_grid=self.param_grid,
                thresholds=[0.5],
                scoring=__scoring,
                refit=_refit,
                # scheduler=scheduler   # pizza
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
        return self.optimal_params_est.visualize(filename=filename, format=format)

    def fit(self, X, y):
        return self.optimal_params_est.fit(X, y)


# END FUNCTION TO BUILD GRIDSEARCH INSTANCES AND ACCESS OUTPUT ** ** ** ** ** *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **





# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** ** ** *

def refit_fxn(cv_results_):
    DF = pd.DataFrame(cv_results_)
    # try: print(DF['rank_test_score'])
    # except: pass
    # try: print(DF['rank_test_balanced_accuracy'])
    # except: pass
    # return 0
    return DF.index[DF['rank_test_balanced_accuracy']==1][0]


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
                print(itr)

                if 'sklearn' in _gscv_type:
                    for _ in (sk_X_train, sk_y_train, sk_X_test, sk_y_test):
                        assert isinstance(_, (np.ndarray, pd.core.frame.DataFrame))

                elif 'dask' in _gscv_type:
                    for _ in (da_X_train, da_y_train, da_X_test, da_y_test):
                        assert isinstance(_, (da.core.Array, ddf.core.DataFrame))

                if 'dask' in _gscv_type:

                    if _gscv_type == 'dask' and _round == 'post_fit' and refit_words == 'refit_fxn':
                        _msg = 'dask cannot take callable for refit'

                        if not dump_to_file:
                            print(f'{itr}: {_msg}')
                        else:
                            ATTR_ARRAY_DICT[itr] = [f'{_msg}' for _ in ATTR_NAMES]
                            METHOD_ARRAY_DICT[itr] = [f'{_msg}' for _ in METHOD_NAMES]

                        continue

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

    # pizza
    scheduler = Client(n_workers=4, threads_per_worker=1)

    with scheduler:
        # WHEN PRINTING TO SCREEN, KEEP access_attrs & access_methods SEPARATE SO THAT
        # attr/method OUTPUT ISNT INTERMINGLED ** ** ** ** ** **

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

        if os.name == 'posix':
            attr_path = rf'/home/bear/Desktop/gscv_attr_comparison_dump.ods'
        elif os.name == 'nt':
            attr_path = rf'c:\users\bill\desktop\gscv_attr_comparison_dump.csv'

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

        if os.name == 'posix':
            method_path = rf'/home/bear/Desktop/gscv_method_comparison_dump.ods'
        elif os.name == 'nt':
            method_path = rf'c:\users\bill\desktop\gscv_method_comparison_dump.csv'

        SINGLE_DF.to_csv(method_path, index=True)
        # END WRITE METHOD RESULTS TO FILE ########################################
        ###########################################################################

# END DASK, SKLEARN, _GSTCV RESPONSES TO ATTR & METHOD CALLS ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **























