# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import inspect


from ._is_dask_estimator import _is_dask_estimator
from base import is_classifier as pybear_is_classifier


def _val_estimator(
    _estimator
    ) -> bool:

    # must be an instance not the class!
    if inspect.isclass(_estimator):
        raise TypeError(f"must be an instance, not the class")

    # must have the sklearn / dask API
    _has_method = lambda _method: callable(getattr(_estimator, _method, None))

    if not _has_method('fit'):
        raise AttributeError(f"estimator must have a 'fit' method")
    if not _has_method('set_params'):
        raise AttributeError(f"estimator must have a 'set_params' method")
    if not _has_method('get_params'):
        raise AttributeError(f"estimator must have a 'get_params' method")
    if not _has_method('score'):
        raise AttributeError(f"estimator must have a 'score' method")

    del _has_method

    # set self._dask_estimator
    __is_dask_estimator = _is_dask_estimator(_estimator)

    if not pybear_is_classifier(_estimator):
        raise TypeError(f"estimator must be a classifier to use threshold. "
            f"use regular sklearn/dask GridSearch CV for a regressor")


    # pizza, temporarily get _is_dask_estimator here and return, hopefully
    # this can be fettered out and made to happen only in main GSTCV
    return __is_dask_estimator







