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

from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from xgboost import XGBClassifier as sk_XGBClassifier

import dask.array as da
import dask.dataframe as ddf
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from xgboost.dask import DaskXGBClassifier as dask_XGBClassifier

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


# THIS MODULE CAPTURES THE OUTPUT (VALUES OR EXC INFO) OF METHOD CALLS
# TO SK GSCV, DASK GSCV, GSTCV, GSTCVDask AFTER fit()

# score IS SEPARATE FROM ALL THE OTHER METHODS BECAUSE ITS SIGNATURE IS
# X, y, BUT ALL THE OTHERS IS JUST X.






# X, y
# score(X, y)


# X
# decision_function(X)
# inverse_transform(Xt)
# predict(X)
# predict_log_proba(X)
# predict_proba(X)
# score_samples(X)
# transform(X)


_rows, _cols = 1000, 10
_bad_cols = 2 * _cols
GOOD_COLUMNS = list(string.ascii_lowercase[:_cols])
BAD_COLUMNS = list(string.ascii_lowercase[:_bad_cols])

good_np_X, good_np_y = sklearn_make_classification(n_samples=_rows,
                                                   n_features=_cols,
                                                   n_informative=_cols,
                                                   n_redundant=0)
good_da_X, good_da_y = da.array(good_np_X), da.array(good_np_y)

good_pd_X, good_pd_y = pd.DataFrame(good_np_X, columns=GOOD_COLUMNS), pd.DataFrame(good_np_y, columns=['y'])
good_ddf_X, good_ddf_y = ddf.from_pandas(good_pd_X, chunksize=_rows // 10), ddf.from_pandas(good_pd_y, chunksize=_rows // 10)

bad_np_X, bad_features_np_y = sklearn_make_classification(n_samples=_rows, n_features=_bad_cols, n_informative=_bad_cols, n_redundant=0)
bad_features_np_y = np.vstack((bad_features_np_y, bad_features_np_y)).reshape(
    (-1, 2))
bad_da_X, bad_features_da_y = da.array(bad_np_X), da.array(bad_features_np_y)

bad_pd_X, bad_features_pd_y = pd.DataFrame(bad_np_X, columns=BAD_COLUMNS), pd.DataFrame(bad_features_np_y, columns=['y1', 'y2'])
bad_ddf_X, bad_features_ddf_y = ddf.from_pandas(bad_pd_X, chunksize=_rows // 10), ddf.from_pandas(bad_features_pd_y, chunksize=_rows // 10)

bad_classes_np_y = np.random.randint(0, 3, _rows)
bad_classes_da_y = da.array(bad_classes_np_y)
bad_classes_pd_y = pd.DataFrame(bad_classes_np_y, columns=['y'])
bad_classes_ddf_y = ddf.from_pandas(bad_classes_pd_y, chunksize=_rows // 10)

GOOD_OR_BAD_X = ['good_X', 'bad_X']
GOOD_OR_BAD_y = ['good_y', 'bad_features_y', 'bad_classes_y']
DTYPES = ['array', 'dataframe']
TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

METHOD_NAMES = [
    'n_features_in_',
    'feature_names_in_',
    'classes_',

    'decision_function',
    'inverse_transform',
    'predict',
    'predict_log_proba',
    'predict_proba',
    'score_samples',
    'transform',

    'score'
]

__scoring = ['balanced_accuracy', 'accuracy']  # 'balanced_accuracy'
__refit = 'balanced_accuracy'

sk_XGB = sk_XGBClassifier()

dask_XGB = dask_XGBClassifier()


def init_gscv(_sk_est, _da_est, _gscv_type):

    _param_grid = {'max_depth': [3,4,5]}

    if _gscv_type == 'sklearn':
        _gscv = sklearn_GridSearchCV(
            estimator=_sk_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score=np.nan,
            return_train_score=True,
            n_jobs=-1
        )

    elif _gscv_type == 'dask':
        _gscv = dask_GridSearchCV(
            estimator=_da_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score=np.nan,
            return_train_score=True,
            n_jobs=-1
        )

    elif _gscv_type == 'gstcv_sklearn':
        _gscv = GSTCV(
            estimator=_sk_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score=np.nan,
            return_train_score=True,
            n_jobs=-1

        )

    elif _gscv_type == 'gstcv_dask':
        _gscv = GSTCVDask(
            estimator=_da_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score=np.nan,
            return_train_score=True,
            n_jobs=-1
        )

    else:
        raise ValueError(f"init_gscv() --- bad gscv_type '{_gscv_type}'")

    return _gscv


def method_output_try_handler(_trial, method_name, _method_output,
                              _METHOD_ARRAY_DICT):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(
            f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")

    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _method_output
    return _METHOD_ARRAY_DICT


def method_output_except_handler(_trial, method_name, _exc_info,
                                 _METHOD_ARRAY_DICT):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(
            f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")
    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _exc_info
    return _METHOD_ARRAY_DICT


COMBINATIONS = [f'{c}_{b}_{a}' for c in GOOD_OR_BAD_X for b in DTYPES for a in TYPES]
METHOD_ARRAY_DICT = {
    k: pd.DataFrame(index=METHOD_NAMES, columns=['OUTPUT'], dtype=object) for k
    in COMBINATIONS}







if __name__ == '__main__':


    ctr = 0
    for good_or_bad_x in GOOD_OR_BAD_X:
        for _dtype in DTYPES:
            for _gscv_type in TYPES:

                ctr += 1
                trial = f'{good_or_bad_x}_{_dtype}_{_gscv_type}'

                print(f'Running {ctr} of {len(COMBINATIONS)}... {trial}')

                test_cls = init_gscv(sk_XGB, dask_XGB, _gscv_type)

                if _dtype == 'array':
                    if 'dask' in _gscv_type:
                        base_y = good_da_y
                        base_X = good_da_X
                        if 'good' in good_or_bad_x:
                            _X = good_da_X
                        elif 'bad' in good_or_bad_x:
                            _X = bad_da_X
                        else:
                            raise Exception(f"good_or_bad_x logic is failing")
                    elif 'sklearn' in _gscv_type:
                        base_y = good_np_y
                        base_X = good_np_X
                        if 'good' in good_or_bad_x:
                            _X = good_np_X
                        elif 'bad' in good_or_bad_x:
                            _X = bad_np_X
                        else:
                            raise Exception(f"good_or_bad_x logic is failing")
                    else:
                        raise Exception(f"_gscv_type logic is failing")
                elif _dtype == 'dataframe':
                    if 'dask' in _gscv_type:
                        base_y = good_np_y  # good_ddf_y  UNPREDICABLE BEHAVIOR PASSING y AS DF TO DASK GridSearcH
                        base_X = good_ddf_X
                        if 'good' in good_or_bad_x:
                            _X = good_ddf_X
                        elif 'bad' in good_or_bad_x:
                            _X = bad_ddf_X
                        else:
                            raise Exception(f"good_or_bad_x logic is failing")
                    elif 'sklearn' in _gscv_type:
                        base_y = good_pd_y
                        base_X = good_pd_X
                        if 'good' in good_or_bad_x:
                            _X = good_pd_X
                        elif 'bad' in good_or_bad_x:
                            _X = bad_pd_X
                        else:
                            raise Exception(f"good_or_bad_x logic is failing")
                    else:
                        raise Exception(f"_gscv_type logic is failing")
                else:
                    raise Exception(f"_dtype logic is failing")

                print(f'trial = {trial}')


                try:
                    test_cls.fit(base_X, base_y)
                except TypeError as e:
                    for _m in METHOD_NAMES:
                        METHOD_ARRAY_DICT[trial].loc[_m, 'OUTPUT'] = e
                    del test_cls, base_X, base_y
                    continue
                except Exception as e2:
                    print(f"\033[91mExcepted for a reason other than "
                          f"dask.dataframe into dask logistic TypeError\033[0m")
                    raise Exception(e2)

                del base_X, base_y

                # ALL METHODS EXCEPT score ** * ** * ** * ** * ** * ** * ** * ** *
                for method in ['decision_function', 'inverse_transform', 'predict',
                               'predict_log_proba', 'predict_proba', 'score_samples',
                               'transform']:

                    try:
                        try:
                            __ = getattr(test_cls, method)(_X).compute()
                        except:
                            __ = getattr(test_cls, method)(_X)

                        METHOD_ARRAY_DICT = method_output_try_handler(
                            trial,
                            method,
                            __,
                            METHOD_ARRAY_DICT
                        )
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(
                            trial,
                            method,
                            sys.exc_info()[1],
                            METHOD_ARRAY_DICT
                        )
                # END ALL METHODS EXCEPT score ** * ** * ** * ** * ** * ** * **

                # REFERENCE ATTRS #############################################
                for attr in ['n_features_in_', 'feature_names_in_', 'classes_']:

                    try:
                        try:
                            __ = getattr(test_cls, attr).compute()
                        except:
                            __ = getattr(test_cls, attr)

                        METHOD_ARRAY_DICT = method_output_try_handler(
                            trial,
                            attr,
                            __,
                            METHOD_ARRAY_DICT
                        )
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(
                            trial,
                            attr,
                            sys.exc_info()[1],
                            METHOD_ARRAY_DICT
                        )
                # END REFERENCE ATTRS #########################################

                del test_cls


SINGLE_DF = pd.DataFrame(index=METHOD_NAMES,
                         columns=list(METHOD_ARRAY_DICT.keys()),
                         dtype='<U100').fillna('-')

for _key, DATA_DF in METHOD_ARRAY_DICT.items():
    SINGLE_DF.loc[:, _key] = DATA_DF.to_numpy().ravel()

SINGLE_DF = SINGLE_DF.drop(['score'], inplace=False)

SINGLE_DF = SINGLE_DF.T



if os.name == 'posix':
    method_path = rf'/home/bear/Desktop/gscv_bad_X_comparison_dump__all_except_score.ods'
elif os.name == 'nt':
    method_path = rf'c:\users\bill\desktop\gscv_bad_X_comparison_dump__all_except_score.csv'

SINGLE_DF.to_csv(method_path, index=True)


# ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## #
# ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## #
# ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## #



def method_output_try_handler(
        _trial,
        method_name,
        _method_output,
        _METHOD_ARRAY_DICT
):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")
    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _method_output
    return _METHOD_ARRAY_DICT


def method_output_except_handler(
        _trial,
        method_name,
        _exc_info,
        _METHOD_ARRAY_DICT
):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")

    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _exc_info
    return _METHOD_ARRAY_DICT


COMBINATIONS = [f'{d}_{c}_{b}_{a}' for d in GOOD_OR_BAD_X for c in
                GOOD_OR_BAD_y for b in DTYPES for a in TYPES]
METHOD_ARRAY_DICT = {
    k: pd.DataFrame(index=METHOD_NAMES, columns=['OUTPUT'], dtype=object) for k
    in COMBINATIONS}

if __name__ == '__main__':
    ctr = 0
    for good_or_bad_x in GOOD_OR_BAD_X:
        for good_or_bad_y in GOOD_OR_BAD_y:
            for _dtype in DTYPES:
                for _gscv_type in TYPES:
                    ctr += 1
                    trial = f'{good_or_bad_x}_{good_or_bad_y}_{_dtype}_{_gscv_type}'

                    print(f'Running {ctr} of {len(COMBINATIONS)}... {trial}')

                    test_cls = init_gscv(sk_XGB, dask_XGB, _gscv_type)

                    if _dtype == 'array':
                        if 'dask' in _gscv_type:
                            base_y = good_da_y
                            if 'good' in good_or_bad_y:
                                _y = good_da_y
                            elif 'bad_features' in good_or_bad_y:
                                _y = bad_features_da_y
                            elif 'bad_classes' in good_or_bad_y:
                                _y = bad_classes_da_y
                            else:
                                raise Exception(f"good_or_bad_y logic is failing")
                            base_X = good_da_X
                            if 'good' in good_or_bad_x:
                                _X = good_da_X
                            elif 'bad' in good_or_bad_x:
                                _X = bad_da_X
                            else:
                                raise Exception(f"good_or_bad_x logic is failing")
                        elif 'sklearn' in _gscv_type:
                            base_y = good_np_y
                            if 'good' in good_or_bad_y:
                                _y = good_np_y
                            elif 'bad_features' in good_or_bad_y:
                                _y = bad_features_np_y
                            elif 'bad_classes' in good_or_bad_y:
                                _y = bad_classes_np_y
                            else:
                                raise Exception(f"good_or_bad_y logic is failing")
                            base_X = good_np_X
                            if 'good' in good_or_bad_x:
                                _X = good_np_X
                            elif 'bad' in good_or_bad_x:
                                _X = bad_np_X
                            else:
                                raise Exception(f"good_or_bad_x logic is failing")
                        else:
                            raise Exception(f"_gscv_type logic is failing")
                    elif _dtype == 'dataframe':
                        if 'dask' in _gscv_type:
                            base_y = good_ddf_y  # UNPREDICABLE BEHAVIOR PASSING y AS DF TO DASK Logistic/GridSearch
                            if 'good' in good_or_bad_y:
                                _y = good_ddf_y
                            elif 'bad_features' in good_or_bad_y:
                                _y = bad_features_ddf_y
                            elif 'bad_classes' in good_or_bad_y:
                                _y = bad_classes_ddf_y
                            else:
                                raise Exception(f"good_or_bad_y logic is failing")
                            base_X = good_ddf_X
                            if 'good' in good_or_bad_x:
                                _X = good_ddf_X
                            elif 'bad' in good_or_bad_x:
                                _X = bad_ddf_X
                            else:
                                raise Exception(f"good_or_bad_x logic is failing")
                        elif 'sklearn' in _gscv_type:
                            base_y = good_pd_y
                            if 'good' in good_or_bad_y:
                                _y = good_pd_y
                            elif 'bad_features' in good_or_bad_y:
                                _y = bad_features_pd_y
                            elif 'bad_classes' in good_or_bad_y:
                                _y = bad_classes_pd_y
                            else:
                                raise Exception(f"good_or_bad_y logic is failing")
                            base_X = good_pd_X
                            if 'good' in good_or_bad_x:
                                _X = good_pd_X
                            elif 'bad' in good_or_bad_x:
                                _X = bad_pd_X
                            else:
                                raise Exception(f"good_or_bad_x logic is failing")
                        else:
                            raise Exception(f"_gscv_type logic is failing")
                    else:
                        raise Exception(f"_dtype logic is failing")

                    try:
                        test_cls.fit(base_X, base_y)
                    except TypeError as e:
                        for _m in METHOD_NAMES:
                            METHOD_ARRAY_DICT[trial].loc[_m, 'OUTPUT'] = e
                        del test_cls, base_X, base_y
                        continue
                    except Exception as e2:
                        print(
                            f"\033[91mExcepted for a reason other than dask.dataframe into dask logistic TypeError\033[0m")
                        raise Exception(e2)

                    del base_X, base_y

                    try:
                        __ = test_cls.score(_X, _y)
                        METHOD_ARRAY_DICT = method_output_try_handler(trial, 'score', __, METHOD_ARRAY_DICT)
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(trial, 'score', sys.exc_info()[1], METHOD_ARRAY_DICT)

                    # REFERENCE ATTRS #################################################################################################
                    try:
                        __ = test_cls.n_features_in_
                        METHOD_ARRAY_DICT = method_output_try_handler(trial, 'n_features_in_', __, METHOD_ARRAY_DICT)
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(trial, 'n_features_in_', sys.exc_info()[1], METHOD_ARRAY_DICT)

                    try:
                        __ = test_cls.feature_names_in_
                        METHOD_ARRAY_DICT = method_output_try_handler(trial, 'feature_names_in_', __, METHOD_ARRAY_DICT)
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(trial, 'feature_names_in_', sys.exc_info()[1], METHOD_ARRAY_DICT)

                    try:
                        __ = test_cls.classes_
                        METHOD_ARRAY_DICT = method_output_try_handler(trial, 'classes_', __, METHOD_ARRAY_DICT)
                    except:
                        METHOD_ARRAY_DICT = method_output_except_handler(trial, 'classes_', sys.exc_info()[1], METHOD_ARRAY_DICT)

SINGLE_DF = pd.DataFrame(index=METHOD_NAMES,
                         columns=list(METHOD_ARRAY_DICT.keys()),
                         dtype='<U100').fillna('-')

for _key, DATA_DF in METHOD_ARRAY_DICT.items():
    SINGLE_DF.loc[:, _key] = DATA_DF.to_numpy().ravel()

SINGLE_DF = SINGLE_DF.drop(
    ['decision_function', 'inverse_transform', 'predict', 'predict_log_proba',
     'predict_proba',
     'score_samples', 'transform'], inplace=False)

SINGLE_DF = SINGLE_DF.T

if os.name == 'posix':
    method_path = rf'/home/bear/Desktop/gscv_bad_X_bad_y_comparison_dump__score.ods'
elif os.name == 'nt':
    method_path = rf'c:\users\bill\desktop\gscv_bad_X_bad_y_comparison_dump__score.csv'

SINGLE_DF.to_csv(method_path, index=True)

# DONE










