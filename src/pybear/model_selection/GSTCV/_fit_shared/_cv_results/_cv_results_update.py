# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numpy as np
import numpy.typing as npt

from ..._type_aliases import (
    CVResultsType,
    IntermediateHolderType,
    ScorerWIPType
)

from ._cv_results_score_updater import _cv_results_score_updater



def _cv_results_update(
    _trial_idx: int,
    _THRESHOLDS: npt.NDArray[Union[int, float]],
    _FOLD_FIT_TIMES_VECTOR: IntermediateHolderType,
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX: IntermediateHolderType,
    _TEST_BEST_THRESHOLD_IDXS_BY_SCORER: npt.NDArray[np.uint16],
    _TEST_FOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _TRAIN_FOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _scorer: ScorerWIPType,
    _cv_results: CVResultsType,
    _return_train_score: bool
) -> CVResultsType:

    """

    Fills a row of cv_results with thresholds, scores, and times, but
    not ranks. (Ranks must be done after cv_results is full.)

    Parameters
    ----------
    _trial_idx:
        int - the row index of cv_results to update
    _THRESHOLDS:
        np.NDArray[Union[int, float]] - vector of thresholds for the
        'param grid' associated with this permutation of search.
        'param grid' being a single dict from the param_grid list of
        param grids.
    _FOLD_FIT_TIMES_VECTOR:
        np.ma.masked_array[float] - the times to fit each of the folds
        for this permutation. If a fit excepted, the corresponding
        position is masked and excluded from aggregate calculations.
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX:
        np.ma.masked_array[float] - A 3D object of shape (n_splits,
        n_thresholds, n_scorers). If a fit excepted, the corresponding
        plane in axis 0 is masked, and is excluded from aggregate
        calculations. Otherwise, holds score times for every fold /
        threshold / scorer permutation.
    _TEST_BEST_THRESHOLD_IDXS_BY_SCORER:
        np.NDArray[np.uint16] - vector of shape (n_scorers,) that matches
        position-for-position against the scorers in scorer_. It holds
        the index location in the original threshold vector of the best
        threshold for each scorer.
    _TEST_FOLD_x_SCORER__SCORE_MATRIX:
        np.ma.masked_array[float] - masked array of shape (n_splits,
        n_scorers) that holds the test scores for the set of folds
        corresponding to the best threshold for that scorer. If a fit
        excepted, the corresponding layer in axis 0 holds 'error_score'
        value in every position.
    _TRAIN_FOLD_x_SCORER__SCORE_MATRIX:
        np.ma.masked_array[float] - masked array of shape (n_splits,
        n_scorers) that holds the train scores for the set of folds
        corresponding to the best threshold for that scorer. If a fit
        excepted, the corresponding layer in axis 0 holds 'error_score'
        value in every position.
    _scorer:
        dict[str, Callable[[Iterable, Iterable], float] -
        dictionary of scorer names and scorer functions. Note that the
        scorer functions are sklearn metrics (or similar), not
        make_scorer. Used to know what column names to look for in
        cv_results and nothing more.
    _cv_results:
        dict[str, np.ma.masked_array] - empty cv_results dictionary other
        than the 'param_' columns and the 'params' column.
    _return_train_score:
        bool - when True, calculate the scores for the train folds in
        addition to the test folds.


    Return
    ------
    -
        _cv_results: dict[str, np.ma.masked_array] - cv_results updated
            with scores, thresholds, and times.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert _trial_idx >= 0, f"'_trial_idx' must be >= 0"
    assert len(_THRESHOLDS) >= 1, f"'_THRESHOLDS must be >= 1 "
    assert isinstance(_scorer, dict), f"'_scorer' must be a dictionary"
    assert all(map(isinstance, _scorer, (str for _ in _scorer))), \
        f"'_scorer' keys must be strings"
    assert all(map(callable, _scorer.values())), \
        f"'_scorer' values must be callables"
    _n_scorers = len(_scorer)
    assert len(_TEST_BEST_THRESHOLD_IDXS_BY_SCORER) == _n_scorers

    assert len(_FOLD_FIT_TIMES_VECTOR) == \
            _TEST_FOLD_x_SCORER__SCORE_MATRIX.shape[0] == \
            _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape[0] == \
            _TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape[0], \
            f"disagreement of number of splits"

    _n_splits = len(_FOLD_FIT_TIMES_VECTOR)

    assert _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape == \
           (_n_splits, len(_THRESHOLDS), _n_scorers), \
            f"bad shape _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX"

    assert _TEST_FOLD_x_SCORER__SCORE_MATRIX.shape == (_n_splits, _n_scorers), \
        f"bad _TEST_FOLD_x_SCORER__SCORE_MATRIX shape"
    assert _TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape == (_n_splits, _n_scorers), \
        f"bad _TRAIN_FOLD_x_SCORER__SCORE_MATRIX shape"


    assert isinstance(_cv_results, dict), f"'_cv_results' must be a dictionary"
    assert isinstance(_return_train_score, bool), \
        f"'_return_train_score' must be bool"

    del _n_scorers, _n_splits
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *


    # UPDATE cv_results_ WITH THRESHOLDS ###############################

    for s_idx, scorer in enumerate(_scorer):

        best_threshold_idx = _TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx]
        best_threshold = _THRESHOLDS[best_threshold_idx]

        scorer = '' if len(_scorer) == 1 else f'_{scorer}'
        if f'best_threshold{scorer}' not in _cv_results:
            raise ValueError(
                f"appending threshold scores to a column in cv_results_ "
                f"that doesnt exist but should (best_threshold{scorer})"
            )

        _cv_results[f'best_threshold{scorer}'][_trial_idx] = best_threshold
    # END UPDATE cv_results_ WITH THRESHOLDS ###########################

    # UPDATE cv_results_ WITH SCORES ###################################
    _cv_results = _cv_results_score_updater(
        _TEST_FOLD_x_SCORER__SCORE_MATRIX,
        'test',
        _trial_idx,
        _scorer,
        _cv_results
    )

    if _return_train_score:
        _cv_results = _cv_results_score_updater(
            _TRAIN_FOLD_x_SCORER__SCORE_MATRIX,
            'train',
            _trial_idx,
            _scorer,
            _cv_results
        )
    # END UPDATE cv_results_ WITH SCORES ###############################



    # UPDATE cv_results_ WITH TIMES ####################################
    for cv_results_column_name in ['mean_fit_time', 'std_fit_time',
                                   'mean_score_time', 'std_score_time']:
        if cv_results_column_name not in _cv_results:
            raise ValueError(
                f"appending time results to a column in cv_results_ that "
                f"doesnt exist but should ({cv_results_column_name})"
            )

    _cv_results['mean_fit_time'][_trial_idx] = np.mean(_FOLD_FIT_TIMES_VECTOR)
    _cv_results['std_fit_time'][_trial_idx] = np.std(_FOLD_FIT_TIMES_VECTOR)

    _cv_results['mean_score_time'][_trial_idx] = \
        np.mean(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
    _cv_results['std_score_time'][_trial_idx] = \
        np.std(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
    # END UPDATE cv_results_ WITH TIMES ################################


    return _cv_results



















