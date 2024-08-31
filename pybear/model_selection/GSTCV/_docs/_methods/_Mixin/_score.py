# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




"""

Score the given X and y using the best estimator, best threshold,
and the defined scorer. When there is only one scorer, that is
the defined scorer, and score is available if refit is not False.
When there are multiple scorers, the defined scorer is the scorer
specified by 'refit', and score is available only if refit is set
to a string.

See the documentation for the 'scoring' parameter for information
about passing kwargs to the scorer.


Parameters
----------
X:
    Iterable[Iterable[Union[int, float]]], shape (n_samples,
    n_features) - Must contain all numerics. Must be able to
    convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
    (GSTCVDask). Must fulfill the input assumptions of the
    underlying estimator.

y:
    Iterable[Union[int, float]], shape (n_samples, ) or
    (n_samples, 1) - The target relative to X. Must be binary in
    [0, 1]. Must be able to convert to a numpy.ndarray (GSTCV)
    or dask.array.core.Array (GSTCVDask).


Return
------
-
    score:
        float - The score for X and y on the best estimator and
        best threshold using the defined scorer.

"""


