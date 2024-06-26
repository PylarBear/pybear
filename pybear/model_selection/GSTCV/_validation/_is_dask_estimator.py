# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import inspect


def _is_dask_estimator(
        _estimator
    ) -> bool:

    # must be an instance not the class!
    if inspect.isclass(_estimator):
        raise TypeError(f"must be an instance, not the class")

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


    _type = str(type(_estimator)).lower()

    # if is a pipeline, pull the last thing in the pipe and call that estimator
    if 'pipe' in _type:
        _estimator = _estimator.steps[-1][-1]


    return 'dask' in str(type(_estimator)).lower()






















