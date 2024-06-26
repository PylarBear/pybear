# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import importlib
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier as sk_is_classifier


def is_classifier(estimator_) -> bool:

    """
    Return True if the given estimator is a classifier, False otherwise. Works
    with scikit-learn, dask_ml, xgboost, and lightgbm estimators. Also works
    for wrapped estimators.

    Support for scikit-learn wrappers includes, but may not be limited to:
        CalibratedClassifierCV\\\n
        GridSearchCV\\\n
        Pipeline\\\n
    Support for dask_ml wrappers includes, but may not be limited to:
        GridSearchCV\\\n
        Incremental\\\n
        ParallelPostFit\\\n
        BlockwiseVotingClassifier\\\n
        BlockwiseVotingRegressor\\\n

    Parameters
    ----------
    estimator_:
        scikit-learn, dask_ml, xgboost, or lightgbm estimator to tests.

    Return
    ------
    -
        bool: True if the estimator is a classifier, False otherwise.

    See Also
    --------
    sklearn.base.is_classifier

    Notes
    -----
    Also supports proper handling of non-estimator objects, returning False
    without raising exception.

    Examples
    --------
    >>> from pybear.base import is_classifier as pybear_is_classifier
    >>> from dask_ml.linear_model import LogisticRegression, LinearRegression
    >>> dask_clf = LogisticRegression()
    >>> is_classifier(dask_clf)
    True
    >>> dask_reg = LinearRegression()
    >>> pybear_is_classifier(dask_reg)
    False

    """



    if sk_is_classifier(estimator_):
        return True

    # USE RECURSION TO GET INNERMOST ESTIMATOR ** ** ** ** ** ** ** ** ** ** **
    recursion_ct = 0

    def retrieve_core_estimator_recursive(_estimator_, recursion_ct):

        recursion_ct += 1
        if recursion_ct == 10: raise Exception(f"too many recursions, abort")

        if 'delayed.delayed' in str(type(_estimator_)).lower():
            return _estimator_

        for _module in [
                        'blockwisevoting',
                        'calibratedclassifier',
                        'gradientboosting'
            ]:

            if str(_estimator_).lower()[:len(_module)] == _module:
                # escape when have dug deep enough that _module is the
                # outermost wrapper. use hard strings, dont import any dask
                # modules to avoid circular imports
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

    # END USE RECURSION TO GET INNERMOST ESTIMATOR ** ** ** ** ** ** ** ** ** *


    if sk_is_classifier(estimator_):
        return True

    if 'blockwisevoting' in str(estimator_).lower():
        # use hard strings, dont import any dask modules to avoid circular imports
        return sk_is_classifier(estimator_)

    try:
        _path = str(estimator_().__class__)
    except:
        _path = str(estimator_.__class__)

    dask_supported = ['lightgbm', 'xgboost', 'dask.delayed', 'dask.array',
                      'dask.dataframe', 'dask_expr._collection.DataFrame']

    if 'dask' in _path and all([x not in _path for x in dask_supported]):
        # use hard strings, dont import any dask modules to avoid circular imports
        _path = _path[_path.find("'", 0, -1) + 1:_path.find("'", -1, 0) - 1]

        _split = _path.split(sep='.')
        if len(_split) == 4: _split.pop(-2)

        _split[0] = 'sklearn'

        _package = ".".join(_split[:2])

        if _split[-1] == 'PoissonRegression':
            _function = 'PoissonRegressor'
        else:
            _function = _split[-1]

        _base_err_msg = f"is_classifier() excepted trying to import "
        try:
            sklearn_module = importlib.import_module(_package)
        except:
            raise ImportError(_base_err_msg + f"{_package}")
        try:
            sklearn_dummy_function = getattr(sklearn_module, _function)
        except:
            raise ImportError(_base_err_msg + f"{_function} from {_package}")

        return sk_is_classifier(sklearn_dummy_function)

    return False









