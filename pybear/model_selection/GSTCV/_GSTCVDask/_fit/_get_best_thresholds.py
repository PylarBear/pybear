# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union, Iterable

import numpy as np
import numpy.typing as npt

from model_selection.GSTCV._type_aliases import IntermediateHolderType, CVResultsType
from model_selection.GSTCV._master_scorer_dict import master_scorer_dict

def _get_best_thresholds(
    _trial_idx: int,
    _scorer_names: list[str],
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _THRESHOLDS: npt.NDArray[Union[float, int]],
    _cv_results: CVResultsType
    ) -> npt.NDArray[np.uint16]:


    err_msg = f"'_trial_idx' must be an integer >= 0"
    try:
        float(_trial_idx)
        if isinstance(_trial_idx, bool):
            raise Exception
        if int(_trial_idx) != _trial_idx:
            raise Exception
        if not _trial_idx >= 0:
            raise Exception
    except:
        raise TypeError(err_msg)
    del err_msg

    err_msg = f"'_scorer_names' must be a list-like of strings"
    try:
        iter(_scorer_names)
        if isinstance(_scorer_names, (str, dict)):
            raise Exception
        for _ in _scorer_names:
            if _ not in master_scorer_dict:
                raise Exception
    except:
        raise ValueError(err_msg)
    del err_msg

    assert isinstance(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                      np.ma.masked_array)
    assert len(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape) == 3, \
        f"'_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX' must be 3D"

    TEST_BEST_THRESHOLD_IDXS_BY_SCORER = \
        np.empty(len(_scorer_names), dtype=np.uint16)

    for s_idx, scorer in enumerate(_scorer_names):

        _SCORER_THRESH_MEANS = \
            _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:, :, s_idx].mean(axis=0)

        _SCORER_THRESH_MEANS = _SCORER_THRESH_MEANS.ravel()

        assert len(_SCORER_THRESH_MEANS) == len(_THRESHOLDS), \
            f"len(_SCORER_THRESH_MEANS) != len(_THRESHOLDS)"

        # IF MULTIPLE THRESHOLDS HAVE BEST SCORE, USE THE ONE CLOSEST TO 0.50
        # FIND CLOSEST TO 0.50 USING (THRESH - 0.50)**2
        BEST_SCORE_IDX_MASK = (_SCORER_THRESH_MEANS == _SCORER_THRESH_MEANS.max())
        del _SCORER_THRESH_MEANS

        MASKED_LSQ = (1 - np.power(_THRESHOLDS - 0.50, 2, dtype=np.float64))
        MASKED_LSQ = MASKED_LSQ * BEST_SCORE_IDX_MASK
        del BEST_SCORE_IDX_MASK

        BEST_LSQ_MASK = (MASKED_LSQ == MASKED_LSQ.max())
        del MASKED_LSQ

        assert len(BEST_LSQ_MASK) == len(_THRESHOLDS), \
            f"len(BEST_LSQ_MASK) != len(THRESHOLDS)"

        best_idx = np.arange(len(_THRESHOLDS))[BEST_LSQ_MASK][0]
        del BEST_LSQ_MASK

        assert int(best_idx) == best_idx, \
            f"int(best_idx) != best_idx"
        assert best_idx in range(len(_THRESHOLDS)), \
            f"best_idx not in range(len(THRESHOLDS))"

        best_threshold = _THRESHOLDS[best_idx]

        scorer = '' if len(_scorer_names) == 1 else f'_{scorer}'
        if f'best_threshold{scorer}' not in _cv_results:
            raise ValueError(f"appending threshold scores to a column in "
                f"cv_results_ that doesnt exist but should (best_threshold{scorer})")

        _cv_results[f'best_threshold{scorer}'][_trial_idx] = best_threshold

        TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx] = best_idx

    del best_idx, best_threshold


    return TEST_BEST_THRESHOLD_IDXS_BY_SCORER










