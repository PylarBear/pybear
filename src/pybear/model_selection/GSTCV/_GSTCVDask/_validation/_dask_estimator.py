# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import ClassifierProtocol

import sys
import inspect
import warnings

from sklearn.pipeline import Pipeline

from .....utilities._check_pipeline import check_pipeline



def _validate_dask_estimator(
    _estimator: ClassifierProtocol
) -> None:

    """
    The GSTCVDask module is expected to most likely encounter dask_ml,
    xgboost, and lightgbm dask estimators. The estimator must be passed
    as an instance, not the class itself.

    Validate that an estimator:
    1) if in a pipe, the pipe is built correctly
    2) warn if it is not a dask classifier, either from dask itself,
    or from XGBoost or LightGBM.
    3) is a classifier, as indicated by the presence of a predict_proba
    method. (early in dev this was done by pybear.base.is_classifier)
    4) meets the other requirements of dask GridSearchCV in having 'fit',
    'set_params', and 'get_params' methods.


    Parameters
    ----------
    _estimator:
        the estimator to be validated


    Return
    ------
    -
        None.


    """


    # must be an instance not the class!
    if inspect.isclass(_estimator):
        raise TypeError(f"estimator must be an instance, not the class")

    # could be a pipeline

    # validate pipeline ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # because sklearn/dask dont do this, and could be hard to detect
    if 'pipe' in str(type(_estimator)).lower():
        check_pipeline(_estimator)
    # END validate pipeline ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # validate estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # if pipe, dig out the estimator ...---...---...---...---...---...---
    def get_inner_most_estimator(__estimator):

        try:
            if isinstance(__estimator, Pipeline):
                return get_inner_most_estimator(__estimator.steps[-1][-1])
            else:
                return get_inner_most_estimator(__estimator.estimator)
        except:
            return __estimator


    __estimator = get_inner_most_estimator(_estimator)

    try:
        _module = sys.modules[__estimator.__class__.__module__].__file__
    except:
        raise AttributeError(f"'{__estimator.__class__.__name__}' is not "
            f"a valid classifier")

    # 24_08_04_07_28_00 change raise to warn
    # to allow XGBClassifier, reference errors associated with
    # DaskXGBClassifier and dask GridSearch CV
    __ = str(_module).lower()
    if 'dask' not in __ and 'conftest' not in __:  # allow pytest with mock clf
        warnings.warn(f"'{__estimator.__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        # raise TypeError(f"'{__estimator.__class__.__name__}' is not a dask "
        #     f"classifier. GSTCVDask can only accept dask classifiers. "
        #     f"\nTo use non-dask classifiers, use the GSTCV package.")

    del get_inner_most_estimator, __estimator, _module

    # END if pipe, dig out the estimator ...---...---...---...---...---.

    # must have the sklearn / dask API
    _has_method = lambda _method: callable(getattr(_estimator, _method, None))

    if not _has_method('fit'):
        raise AttributeError(f"estimator must have a 'fit' method")
    if not _has_method('set_params'):
        raise AttributeError(f"estimator must have a 'set_params' method")
    if not _has_method('get_params'):
        raise AttributeError(f"estimator must have a 'get_params' method")
    if not _has_method('predict_proba'):
        raise AttributeError(f"estimator must have a 'predict_proba' method")

    del _has_method
    # END validate estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    return





