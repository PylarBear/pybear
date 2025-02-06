# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numpy as np
import numpy.typing as npt

from .._type_aliases import IntermediateHolderType



def _get_best_thresholds(
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _THRESHOLDS: npt.NDArray[Union[float, int]]
    ) -> npt.NDArray[np.uint16]:

    """

    After collecting the scores for every fold / threshold / scorer
    combination, average the scores across the folds for each scorer to
    give the mean scores of fits in vectors of shape (n_thresholds, ).
    If a fit excepted, every value in the corresponding plane on axis 0
    was set to 'error_score'. If error_score was numeric, that fold
    is included in the mean calculations; if that number was np.nan,
    that fold is excluded from the mean calculations. With the vector of
    mean scores, apply an algorithm that finds the index position of the
    maximum mean score, and if there are multiple positions with that
    value, finds the position that is closest to 0.5. Repeat this for all
    scorers to populate a TEST_BEST_THRESHOLD_IDXS_BY_SCORER vector with
    the index position of the best threshold for each scorer.


    Parameters
    ----------
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX:
        np.ma.masked_array[float] - A 3D object of shape (n_splits,
        n_thresholds, n_scorers). If a fit excepted, the corresponding
        plane in axis 0 holds the 'error_score' value in every position.
        Otherwise, holds scores for every fold / threshold / scorer
        permutation.
    _THRESHOLDS: npt.NDArray[Union[float, int]]
        np.NDArray[Union[int, float]] - vector of thresholds for the
        'param grid' associated with this permutation of search.
        'param grid' being a single dict from the param_grid list of
        param grids.


    Return
    ------
    -
        TEST_BEST_THRESHOLD_IDXS_BY_SCORER: npt.NDArray[np.uint16] -
            A vector of shape (n_scorers, ) that holds the index in the
            'thresholds' vector of the threshold that had the highest
            score (or lowest loss) for each scorer.


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    assert isinstance(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                      np.ma.masked_array)
    assert len(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape) == 3, \
        f"'_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX' must be 3D"
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    TEST_BEST_THRESHOLD_IDXS_BY_SCORER = \
        np.ma.zeros(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape[2],
        dtype=np.uint16
    )

    for s_idx, scorer in enumerate(TEST_BEST_THRESHOLD_IDXS_BY_SCORER):

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

        best_idx = np.argmax(MASKED_LSQ)
        del MASKED_LSQ

        assert int(best_idx) == best_idx, \
            f"int(best_idx) != best_idx"
        assert best_idx in range(len(_THRESHOLDS)), \
            f"best_idx not in range(len(THRESHOLDS))"

        TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx] = best_idx

    del best_idx


    return TEST_BEST_THRESHOLD_IDXS_BY_SCORER










