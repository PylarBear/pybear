# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Iterable, Union

import numpy as np

from ..._type_aliases import (
    CVResultsType,
    IntermediateHolderType,
    ScorerWIPType
)

from ._cv_results_score_updater import _cv_results_score_updater


def _cv_results_update(
    _trial_idx: int,
    _TEST_BEST_THRESHOLD_IDXS_BY_SCORER: IntermediateHolderType,
    _TEST_FOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX: IntermediateHolderType,
    _TEST_FOLD_FIT_TIME_VECTOR: IntermediateHolderType,
    _TRAIN_FOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    _THRESHOLDS: Iterable[Union[int, float]],
    _scorer: ScorerWIPType,
    _n_splits: int,
    _cv_results: CVResultsType,
    _return_train_score: bool
    ) -> CVResultsType:

    # UPDATE cv_results_ WITH THRESHOLDS ##############################

    for s_idx, scorer in enumerate(_scorer):

        best_threshold_idx = _TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx]
        best_threshold = _THRESHOLDS[best_threshold_idx]

        scorer = '' if len(_scorer) == 1 else f'_{scorer}'
        if f'best_threshold{scorer}' not in _cv_results:
            raise ValueError(
                f"appending threshold scores to a column in "
                f"cv_results_ that doesnt exist but should (best_threshold{scorer})")

        _cv_results[f'best_threshold{scorer}'][_trial_idx] = best_threshold
    # END UPDATE cv_results_ WITH THRESHOLDS ##########################

    # UPDATE cv_results_ WITH SCORES ##################################
    _cv_results = _cv_results_score_updater(
        _TEST_FOLD_x_SCORER__SCORE_MATRIX,
        'test',
        _trial_idx,
        _scorer,
        _n_splits,
        _cv_results
    )

    if _return_train_score:
        _cv_results = _cv_results_score_updater(
            _TRAIN_FOLD_x_SCORER__SCORE_MATRIX,
            'train',
            _trial_idx,
            _scorer,
            _n_splits,
            _cv_results
        )
    # END UPDATE cv_results_ WITH SCORES ##############################



    # UPDATE cv_results_ WITH TIMES ############################
    for cv_results_column_name in ['mean_fit_time', 'std_fit_time',
                                   'mean_score_time', 'std_score_time']:
        if cv_results_column_name not in _cv_results:
            raise ValueError(f"appending time results to a column in cv_results_ that doesnt exist but should ({cv_results_column_name})")

    _cv_results['mean_fit_time'][_trial_idx] = np.mean(_TEST_FOLD_FIT_TIME_VECTOR)
    _cv_results['std_fit_time'][_trial_idx] = np.std(_TEST_FOLD_FIT_TIME_VECTOR)

    _cv_results['mean_score_time'][_trial_idx] = np.mean(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
    _cv_results['std_score_time'][_trial_idx] = np.std(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
    # END UPDATE cv_results_ WITH TIMES ########################


    return _cv_results



















