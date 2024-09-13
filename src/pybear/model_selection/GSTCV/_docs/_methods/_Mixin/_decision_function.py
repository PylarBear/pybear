# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

Call decision_function on the estimator with the best found
parameters. Only available if refit is not False and the
underlying estimator supports decision_function.


Parameters
----------
X:
    Iterable[Iterable[Union[int, float]]], shape (n_samples,
    n_features) - Must contain all numerics. Must be able to
    convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
    (GSTCVDask). Must fulfill the input assumptions of the
    underlying estimator.


Return
------
-
    The best_estimator_ decision_function method result for X.

"""


