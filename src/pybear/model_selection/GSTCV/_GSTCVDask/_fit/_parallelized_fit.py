# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import (
    XDaskWIPType,
    YDaskWIPType,
    ClassifierProtocol
)

import time
import warnings

import dask



def _parallelized_fit(
    f_idx: int,
    _X_train: XDaskWIPType,
    _y_train: YDaskWIPType,
    _estimator_: ClassifierProtocol,
    _grid: dict[str, Union[str, int, float, bool]],
    _error_score,
    **fit_params
) -> tuple[ClassifierProtocol, float, bool]:

    """
    Estimator fit method designed for dask parallelism. Special dask_ml
    GSCV exception handling on fit.


    Parameters
    ----------
    f_idx:
        int - the zero-based split index of the train partition used in
        this fit; parallelism occurs over the different splits.
    _X_train:
        dask.array.core.Array[Union[int,float]] - A train partition of
        the data being fit. Must be 2D da.core.Array.
    _y_train:
        dask.array.core.Array[int] - The corresponding train partition
        of the target for the X train partition. Must be 1D da.core.Array.
    _estimator_:
        ClassifierProtocol - Any classifier that fulfills the dask_ml
        API for classifiers, having fit, predict_proba, get_params, and
        set_params methods (the score method is not necessary, as GSTCV
        never calls it.) This includes, but is not limited to, dask_ml,
        XGBoost, and LGBM classifiers.
    _grid:
        dict[str, Union[str, int, float, bool]] - the hyperparameter
        values to be used during this fit. One permutation of all the
        grid search permutations.
    _error_score:
        Union[int, float, Literal['raise']] - if a training fold excepts
        during fitting, the exception can be allowed to raise by passing
        the 'raise' literal. Otherwise, passing a number or number-like
        will cause the exception to be handled, allowing the grid search
        to proceed, and the given number carries through scoring
        tabulations in place of the missing scores.
    **fit_params:
        **dict[str, any] - dictionary of kwarg: value pairs to be passed
        to the estimator's fit method.


    Return
    ------
    -
        _estimator_:
            EstimatorProtocol - the fit estimator
        _fit_time:
            float - the time required to perform the fit
        _fit_excepted:
            bool - True if the fit excepted and '_error_score' was not
            'raise'; False if the fit ran successfully.


    """



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    assert isinstance(f_idx, int)
    assert isinstance(_X_train, dask.array.core.Array)
    assert isinstance(_y_train, dask.array.core.Array)
    assert isinstance(_grid, dict)
    assert isinstance(_error_score, (str, float, int))

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *


    t0_fit = time.perf_counter()

    fit_excepted = False

    _X_train = _X_train.persist()
    _y_train = _y_train.persist()

    try:
        _estimator_.fit(_X_train, _y_train, **fit_params)
    except BrokenPipeError:
        raise BrokenPipeError  # FOR PYTEST ONLY
    except Exception as f:
        if _error_score == 'raise':
            raise ValueError(f"estimator excepted during fitting on {_grid}, "
                                f"cv fold index {f_idx} --- {f}")
        else:
            fit_excepted = True
            warnings.warn(
                f'\033[93mfit excepted on {_grid}, cv fold index {f_idx}\033[0m'
            )

    _fit_time = time.perf_counter() - t0_fit

    del t0_fit

    return _estimator_, _fit_time, fit_excepted





