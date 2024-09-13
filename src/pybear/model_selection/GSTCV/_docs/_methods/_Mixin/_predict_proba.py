# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

Call predict_proba on the estimator with the best found
parameters. Only available if refit is not False. The underlying
estimator must support this method, as it is a characteristic
that is validated.


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
    The best_estimator_ predict_proba_ method result for X.

"""


