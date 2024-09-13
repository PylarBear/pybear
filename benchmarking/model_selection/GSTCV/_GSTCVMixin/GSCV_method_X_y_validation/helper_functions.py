# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask




def init_gscv(_sk_est, _da_est, _gscv_type, _param_grid, __scoring, __refit):


    if _gscv_type == 'sklearn':
        _gscv = sklearn_GridSearchCV(
            estimator=_sk_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score='raise',
            return_train_score=False,
            n_jobs=-1
        )

    elif _gscv_type == 'dask':
        _gscv = dask_GridSearchCV(
            estimator=_da_est,
            param_grid=_param_grid,
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score='raise',
            return_train_score=False,
            n_jobs=-1
        )

    elif _gscv_type == 'gstcv_sklearn':
        _gscv = GSTCV(
            estimator=_sk_est,
            param_grid=_param_grid,
            thresholds=[0.5],
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score='raise',
            return_train_score=False,
            n_jobs=-1

        )

    elif _gscv_type == 'gstcv_dask':
        _gscv = GSTCVDask(
            estimator=_da_est,
            param_grid=_param_grid,
            thresholds=[0.5],
            scoring=__scoring,
            refit=__refit,
            cv=5,
            error_score='raise',
            return_train_score=False,
            n_jobs=-1,
            scheduler=None
        )

    else:
        raise ValueError(f"init_gscv() --- bad gscv_type '{_gscv_type}'")

    return _gscv




def method_output_try_handler(_trial, method_name, _method_output, _METHOD_ARRAY_DICT):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(
            f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")

    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _method_output
    return _METHOD_ARRAY_DICT


def method_output_except_handler(_trial, method_name, _exc_info, _METHOD_ARRAY_DICT):
    if _trial not in _METHOD_ARRAY_DICT:
        raise ValueError(
            f"trying to modify key {_trial} in METHOD_ARRAY_DICT but key doesnt exist")

    _METHOD_ARRAY_DICT[_trial].loc[method_name, 'OUTPUT'] = _exc_info
    return _METHOD_ARRAY_DICT



