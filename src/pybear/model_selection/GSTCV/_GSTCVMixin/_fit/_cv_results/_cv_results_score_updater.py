# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
from typing import Literal
from ...._type_aliases import CVResultsType, ScorerWIPType



def _cv_results_score_updater(
    _FOLD_x_SCORER__SCORE_MATRIX: np.ma.masked_array[float],
    _type: Literal['train', 'test'],
    _trial_idx: int,
    _scorer: ScorerWIPType,
    _cv_results: CVResultsType
) -> CVResultsType:

    """

    This module supports _cv_results_update().

    Update the correct permutation row (_trial_idx) and column
    ({'mean'/'std'/'split'}{_split/''}_{_type}_{scorer/'score'}) of
    cv_results with the scores from _FOLD_x_SCORER__SCORE_MATRIX. The
    _FOLD_x_SCORER__SCORE_MATRIX grid can contain either test scores or
    train scores (not at the same time!)


    Parameters
    ----------
    _FOLD_x_SCORER__SCORE_MATRIX:
        np.ma.masked_array[float] - grid of shape (n splits, n scorers)
        that holds either train scores or test scores.
    _type:
        Literal['train', 'test'] - indicates whether the scores being
        updated are for train or test.
    _trial_idx:
        int - row index of cv_results to update
    _scorer:
        dict[str, Callable[[Iterable, Iterable], float] -
        dictionary of scorer names and scorer functions. Note that
        the scorer functions are sklearn metrics (or similar), not
        make_scorer. Used to know what column names to look for in
        cv_results and nothing more.
    _cv_results:
        dict[str, np.ma.masked_array] - tabulated scores, times, etc.,
        of grid search trials


    Return
    ------
    -
        _cv_results: dict[str, np.ma.masked_array] - cv_results updated
            with scores


    """


    # _validation ** * ** *
    assert _FOLD_x_SCORER__SCORE_MATRIX.shape[0] >= 2

    assert _FOLD_x_SCORER__SCORE_MATRIX.shape[1] >= 1

    if _type not in ('train', 'test'):
        raise ValueError(f"'_type' ({_type}) must be 'train' or 'test'")

    _n_permutes = len(_cv_results[list(_cv_results.keys())[0]])
    if _trial_idx not in range(_n_permutes):
        raise ValueError(f"'_trial_idx' ({_trial_idx}) out of range for "
            f"cv_results with {_n_permutes} permutations")
    del _n_permutes

    # to allow for any user-defined scorer name
    # for _scorer_name in _scorer:
    #     if _scorer_name != 'score' and _scorer_name not in master_scorer_dict:
    #         raise ValueError(f"scorer names in '_scorer' ({_scorer_name}) must "
    #             f"match those in allowed: {', '.join(master_scorer_dict)}")

    assert isinstance(_cv_results, dict)

    # - - - - - - - -
    def no_column_err(_header: str) -> None:

        raise ValueError(
            f"appending scores to a column in cv_results_ that doesnt "
            f"exist ({_header})"
        )
    # - - - - - - - -

    for scorer_idx, scorer_suffix in enumerate(_scorer):

        if len(_scorer) == 1:
            scorer_suffix = 'score'

        # individual splits
        for _split in range(_FOLD_x_SCORER__SCORE_MATRIX.shape[0]):

            _header = f'split{_split}_{_type}_{scorer_suffix}'
            if _header not in _cv_results:
                no_column_err(_header)

        # mean of all splits
        _header = f'mean_{_type}_{scorer_suffix}'
        if _header not in _cv_results:
            no_column_err(_header)

        # stdev of all splits
        _header = f'std_{_type}_{scorer_suffix}'
        if _header not in _cv_results:
            no_column_err(_header)

    del no_column_err, _header
    # END _validation ** * ** *



    for scorer_idx, scorer_suffix in enumerate(_scorer):

        if len(_scorer) == 1:
            scorer_suffix = 'score'

        # individual splits
        for _split in range(_FOLD_x_SCORER__SCORE_MATRIX.shape[0]):

            _cv_results[f'split{_split}_{_type}_{scorer_suffix}'][_trial_idx] = \
                _FOLD_x_SCORER__SCORE_MATRIX[_split, scorer_idx]

        # mean of all splits
        _cv_results[f'mean_{_type}_{scorer_suffix}'][_trial_idx] = \
            np.mean(_FOLD_x_SCORER__SCORE_MATRIX[:, scorer_idx])

        # stdev of all splits
        _cv_results[f'std_{_type}_{scorer_suffix}'][_trial_idx] = \
            np.std(_FOLD_x_SCORER__SCORE_MATRIX[:, scorer_idx])


    return _cv_results

















