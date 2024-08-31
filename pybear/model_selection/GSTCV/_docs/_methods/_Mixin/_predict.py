# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

Call the best estimator's predict_proba method on the passed X
and apply the best_threshold_ to predict the classes for X. When
only one scorer is used, predict is available if refit is not
False. When more than one scorer is used, predict is only
available if refit is set to a string.


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
    A vector in [0,1] indicating the class label for the examples
    in X. A numpy.ndarray (GSTCV) or dask.array.core.Array
    (GSTCVDask) is returned.

"""






