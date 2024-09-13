# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

Call inverse_transform on the estimator with the best found
parameters. Only available if refit is not False and the
underlying estimator supports inverse_transform.


Parameters
----------
X:
    Iterable[Iterable[Union[int, float]]] - Must contain all
    numerics. Must be able to convert to a numpy.ndarray (GSTCV)
    or dask.array.core.Array (GSTCVDask). Must fulfill the input
    assumptions of the underlying estimator.


Return
------
-
    The best_estimator_ inverse_transform method result for X.


"""


