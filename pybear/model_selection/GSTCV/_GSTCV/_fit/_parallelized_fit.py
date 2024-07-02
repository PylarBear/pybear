# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union
import time
import warnings
import joblib

from model_selection.GSTCV._type_aliases import (
    XSKWIPType,
    XDaskWIPType,
    YSKWIPType,
    YDaskWIPType,
    ClassifierProtocol
)



@joblib.wrap_non_picklable_objects
def _parallelized_fit(
        f_idx: int,
        X_train: Union[XSKWIPType, XDaskWIPType],
        y_train: Union[YSKWIPType, YDaskWIPType],
        _estimator_: ClassifierProtocol,
        _grid: dict[str, Union[str, int, float, bool]],
        _error_score,
        **fit_params
    ):


    t0_fit = time.perf_counter()

    fit_excepted = False


    try:
        _estimator_.fit(X_train, y_train, **fit_params)
    except TypeError as e:  # 24_02_27_14_39_00 HANDLE PASSING DFS TO DASK ESTIMATOR
        raise TypeError(e)
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














