# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import time

from typing import Union

import numpy as np
import numpy.typing as npt
import dask

from ....GSTCV._type_aliases import (
    ScorerWIPType,
    ClassifierProtocol,
    XDaskWIPType,
    YDaskWIPType
)


def _parallelized_scorer(
    _X_test: XDaskWIPType,
    _y_test: YDaskWIPType,
    _FIT_OUTPUT_TUPLE: tuple[ClassifierProtocol, float, bool],
    _f_idx: int,
    _SCORER_DICT: ScorerWIPType,
    _THRESHOLDS: npt.NDArray[np.float64],
    _error_score: Union[int, float, None],
    _verbose: int,
    **scorer_params
    ) -> tuple[np.ma.masked_array, np.ma.masked_array]:

    # dont adjust the spacing, is congruent with train scorer

    """
    Using the estimators fit for each train fold, use predict_proba and
    _X_tests to generate _y_preds and score against the corresponding
    _y_tests using all of the scorers.
    Build one fold layer of the TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE and
    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME cubes.


    Parameters
    ----------
    _X_test:
        dask.array.core.Array[Union[int,float]] - A test partition of the
        data, matched up with the estimator that was trained on the
        complementary train set. Must be 2D da.core.Array.
    _y_test:
        dask.array.core.Array[int] - The corresponding test partition of
        the target for the X test partition. Must be 1D da.core.Array.
    _FIT_OUTPUT_TUPLE:
        tuple[ClassifierProtocol, float, bool] - A tuple holding the
        fitted estimator, the fit time (not needed here), and the
        fit_excepted boolean (needed here.)
    _f_idx:
        int - the zero-based split index of the test partition used here;
        parallelism occurs over the different splits.
    _SCORER_DICT:
        dict[str: Callable[[Iterable[int], Iterable[int]], float] -
        a dictionary with scorer name as keys and the scorer callables
        as values. The scorer callables are sklearn metrics (or similar),
        not make_scorer.
    _THRESHOLDS:
        npt.NDArray[np.float64] - for the current search permutation,
        there was a mother param grid that contained a 'thresholds'
        parameter, that was separated from the mother before building
        cv_results. This is the vector of thresholds from the mother that
        also mothered this search permutation.
    _error_score:
        Union[int, float, Literal['raise']] - if the training fold
        complementing this test fold excepted during fitting and
        error_score was set to the 'raise' literal, this module cannot
        be reached. Otherwise, a number or number-like was passed to
        'error_score'. If 'fit_excepted' is True, this module puts the
        'error_score' value in every position of the
        TEST_THRESHOLD_x_SCORER__SCORE_LAYER array. If 'error_score' is
        set to np.nan, that layer is also masked. Every value in
        TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER is set to np.nan and
        masked.
    _verbose:
        int - a number from 0 to 10 that indicates the amount of
        information to display to the screen during the grid search
        process. 0 means no output, 10 means maximum output.
        process. 0 means no output, 10 means maximum output.
    **scorer_params:
        **dict[str: any] - dictionary of kwargs to be passed to the scorer
        metrics. 24_07_13 not used by the calling _core_fit module.


    Return
    ------
    -
        TEST_THRESHOLD_x_SCORER__SCORE_LAYER:
            np.ma.masked_array - masked array of shape (n_thresholds,
            n_scorers
        TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER:
            np.ma.masked_array - masked array of shape (n_thresholds,
            n_scorers


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X_test, dask.array.core.Array)
    assert isinstance(_y_test, dask.array.core.Array)
    assert isinstance(_FIT_OUTPUT_TUPLE, tuple)
    assert isinstance(_f_idx, int)
    assert isinstance(_SCORER_DICT, dict)
    assert all(map(callable, _SCORER_DICT.values()))
    assert isinstance(_THRESHOLDS, np.ndarray)
    assert isinstance(_verbose, (int, float))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *

    if _verbose >= 5:
        print(f"Start scoring fold {_f_idx + 1} test with different thresholds "
              f"and scorers")

    _estimator_, _fit_time, _fit_excepted = _FIT_OUTPUT_TUPLE

    TEST_THRESHOLD_x_SCORER__SCORE_LAYER = \
        np.ma.zeros((len(_THRESHOLDS), len(_SCORER_DICT)), dtype=np.float64)
    TEST_THRESHOLD_x_SCORER__SCORE_LAYER.mask = True

    TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER = \
        np.ma.zeros((len(_THRESHOLDS), len(_SCORER_DICT)), dtype=np.float64)
    TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER.mask = True

    # IF A FOLD EXCEPTED DURING FIT, ALL THE THRESHOLDS AND SCORERS
    # IN THAT FOLD LAYER GET SET TO error_score.
    # SCORE TIME CANT BE TAKEN SINCE SCORING WONT BE DONE SO ALSO MASK THAT
    if _fit_excepted:
        TEST_THRESHOLD_x_SCORER__SCORE_LAYER[:, :] = _error_score
        if _error_score is np.nan:
            TEST_THRESHOLD_x_SCORER__SCORE_LAYER[:, :] = np.ma.masked

        TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER[:, :] = np.nan
        TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER[:, :] = np.ma.masked

        if _verbose >= 5:
            print(f'fold {_f_idx + 1} excepted during fit, unable to score')

        return (TEST_THRESHOLD_x_SCORER__SCORE_LAYER,
                    TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER)

    # v v v only accessible if fit() did not except v v v

    _X_test = _X_test.persist()
    _y_test = _y_test.persist()

    pp0_time = time.perf_counter()
    _predict_proba = _estimator_.predict_proba(_X_test)[:, -1].ravel()
    pp_time = time.perf_counter() - pp0_time
    del pp0_time

    if _verbose >= 5:
        print(f'fold {_f_idx + 1} test predict_proba time = {pp_time: ,.3g} s')
    del pp_time

    # GET SCORES FOR ALL SCORERS & THRESHOLDS ##########################

    _test_fold_score_t0 = time.perf_counter()
    for thresh_idx, _threshold in enumerate(_THRESHOLDS):

        _y_pred_t0 = time.perf_counter()
        _y_test_pred = (_predict_proba >= _threshold).astype(np.uint8)
        _ypt = time.perf_counter() - _y_pred_t0
        del _y_pred_t0
        if _verbose == 10:
            print(f"fold {_f_idx+1} test thresholding time = {_ypt: ,.3g} s")
        del _ypt

        for s_idx, scorer_key in enumerate(_SCORER_DICT):






            test_scorer_t0 = time.perf_counter()
            _score = _SCORER_DICT[scorer_key](_y_test, _y_test_pred, **scorer_params)
            test_scorer_score_time = time.perf_counter() - test_scorer_t0
            del test_scorer_t0
            TEST_THRESHOLD_x_SCORER__SCORE_LAYER[thresh_idx, s_idx] = _score
            TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER[thresh_idx, s_idx] = \
                test_scorer_score_time

            if _verbose >= 8:
                print(f"fold {_f_idx + 1} '{scorer_key}' test score time = "
                      f"{test_scorer_score_time: ,.3g} s")

        del _score, test_scorer_score_time


    tfst = time.perf_counter() - _test_fold_score_t0
    del _test_fold_score_t0

    # END GET SCORE FOR ALL SCORERS & THRESHOLDS ##############################

    if _verbose >= 5:
        print(f'End scoring fold {_f_idx + 1} test with different thresholds and '
              f'scorers')
        _ = _f_idx + 1

        print(f'fold {_} total test thresh & score wall time = {tfst: ,.3g} s')
        _ast = tfst / len(_SCORER_DICT) / len(_THRESHOLDS)
        print(f'fold {_} avg test thresh & score wall time = {_ast: ,.3g} s')
        del _ast

        __ = TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER
        print(f'fold {_} total test actual scoring time = {__.sum(): ,.3g} s')
        print(f'fold {_} avg test actual scoring time = {__.mean(): ,.3g} s')
        del _, __

    del _X_test, _y_test, _predict_proba, _y_test_pred, tfst


    return (TEST_THRESHOLD_x_SCORER__SCORE_LAYER,
                TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER)



