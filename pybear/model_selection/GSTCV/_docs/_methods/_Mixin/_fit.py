# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


"""

Perform the grid search with the hyperparameter settings in
param grid to generate scores for the given X and y.


Parameters
----------
X:
    Iterable[Iterable[Union[int, float]]], shape (n_samples,
    n_features) - The data on which to perform the grid search.
    Must contain all numerics. Must be able to convert to a
    numpy.ndarray (GSTCV) or dask.array.core.Array (GSTCVDask).
    Must fulfill the input assumptions of the underlying
    estimator.

y:
    Iterable[int], shape (n_samples,) or (n_samples, 1) - The
    target relative to X. Must be binary in [0, 1]. Must be able
    to convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
    (GSTCVDask). Must fulfill the input assumptions of the
    underlying estimator.

**params:
    dict[str, any] - Parameters passed to the fit method of the
    estimator. If a fit parameter is an array-like whose length
    is equal to num_samples, then it will be split across CV
    groups along with X and y. For example, the sample_weight
    parameter is split because len(sample_weights) = len(X). For
    array-likes intended to be subject to CV splits, care must
    be taken to ensure that any such vector is shaped
    (num_samples, ) or (num_samples, 1), otherwise it will not
    be split. For GSTCVDask, pybear recommends passing such
    array-likes as dask arrays.

    For pipelines, fit parameters can be passed to the fit method
    of any of the steps. Prefix the parameter name with the name
    of the step, such that parameter p for step s has key s__p.


Return
------
-
    self: fitted estimator instance - GSTCV(Dask) instance.


"""




