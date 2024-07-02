# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union
import time
import joblib
import numpy as np
import numpy.typing as npt
import dask.array as da

from model_selection.GSTCV._type_aliases import YSKWIPType, YDaskWIPType, ScorerWIPType

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict


@joblib.wrap_non_picklable_objects
def _threshold_scorer_sweeper(
        _threshold: Union[float, int],
        _y_test: Union[YSKWIPType, YDaskWIPType],
        _predict_proba: Union[YSKWIPType, YDaskWIPType],
        _SCORER_DICT: ScorerWIPType,
        **scorer_params
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:


    try:
        float(_threshold)
        if isinstance(_threshold, bool):
            raise Exception
        if _threshold < 0 or _threshold > 1:
            raise Exception
    except:
        raise ValueError(f"'_threshold' must be a number 0 <= x <= 1")


    assert isinstance(_SCORER_DICT, dict)
    assert all(map(callable, _SCORER_DICT.values()))
    assert all(map(lambda x: x in master_scorer_dict, _SCORER_DICT))

    assert isinstance(_y_test, (np.ndarray, da.core.Array))
    assert isinstance(_predict_proba, (np.ndarray, da.core.Array))

    # try:
    #     _y_test = _y_test.compute()
    # except:
    #     pass
    #
    # try:
    #     _predict_proba = _predict_proba.compute()
    # except:
    #     pass


    SINGLE_THRESH_AND_FOLD__SCORE_VECTOR = \
        np.empty(len(_SCORER_DICT), dtype=np.float64)

    SINGLE_THRESH_AND_FOLD__TIME_VECTOR = \
        np.empty(len(_SCORER_DICT), dtype=np.float64)

    _y_test_pred = (_predict_proba >= _threshold).astype(np.uint8)

    for s_idx, scorer_key in enumerate(_SCORER_DICT):
        test_t0_score = time.perf_counter()
        __ = _SCORER_DICT[scorer_key](_y_test, _y_test_pred, **scorer_params)
        test_tf_score = time.perf_counter() - test_t0_score
        SINGLE_THRESH_AND_FOLD__TIME_VECTOR[s_idx] = test_tf_score
        SINGLE_THRESH_AND_FOLD__SCORE_VECTOR[s_idx] = __

    del __

    return (SINGLE_THRESH_AND_FOLD__SCORE_VECTOR,
            SINGLE_THRESH_AND_FOLD__TIME_VECTOR)









