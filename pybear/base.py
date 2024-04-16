# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import importlib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier as sk_is_classifier


### CORE dask is_classifier FUNCTION ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

def is_classifier(estimator_):
    if sk_is_classifier(estimator_):
        return True

    # USE RECURSION TO GET INNERMOST ESTIMATOR ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    recursion_ct = 0

    def retrieve_core_estimator_recursive(_estimator_, recursion_ct):

        recursion_ct += 1
        if recursion_ct == 10: raise Exception(f"too many recursions, abort")

        if 'delayed.delayed' in str(type(_estimator_)).lower(): return _estimator_

        for _module in ['blockwisevoting', 'calibratedclassifier', 'gradientboosting']:

            if str(_estimator_).lower()[:len(_module)] == _module:
                # escape when have dug deep enough that _module is the outermost wrapper
                # use hard strings, dont import any dask modules to avoid circular imports
                return _estimator_

        try:
            _estimator_ = _estimator_.estimator
        except:
            if isinstance(_estimator_, Pipeline):
                _estimator_ = _estimator_.steps[-1][-1]
            else:
                return _estimator_

        return retrieve_core_estimator_recursive(_estimator_, recursion_ct)

    estimator_ = retrieve_core_estimator_recursive(estimator_, recursion_ct)
    # END USE RECURSION TO GET INNERMOST ESTIMATOR ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    if sk_is_classifier(estimator_):
        return True

    if 'blockwisevoting' in str(estimator_).lower():
        # use hard strings, dont import any dask modules to avoid circular imports
        return sk_is_classifier(estimator_)

    try:
        _path = str(estimator_().__class__)
    except:
        _path = str(estimator_.__class__)


    if 'dask' in _path and \
        np.array([x not in _path for x in
            ['lightgbm', 'xgboost', 'dask.delayed', 'dask.array', 'dask.dataframe', 'dask_expr._collection.DataFrame']]).all():
        # use hard strings, dont import any dask modules to avoid circular imports
        _path = _path[_path.find("'", 0, -1) + 1:_path.find("'", -1, 0) - 1]

        _split = _path.split(sep='.')
        if len(_split) == 4: _split.pop(-2)

        _split[0] = 'sklearn'

        _package = ".".join(_split[:2])

        _function = 'PoissonRegressor' if _split[-1] == 'PoissonRegression' else _split[-1]

        try:
            sklearn_module = importlib.import_module(_package)
        except:
            raise Exception(f"is_classifier() excepted trying to import {_package}")
        try:
            sklearn_dummy_function = getattr(sklearn_module, _function)
        except:
            raise Exception(f"is_classifier() excepted trying to import {_function} from {_package}")

        return sk_is_classifier(sklearn_dummy_function)

    return False




### END CORE dask is_classifier FUNCTION ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


