# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Literal
import time


import numpy as np
import numpy.typing as npt

from ..._type_aliases import (
    CVResultsType,
    ClassifierProtocol,
    ScorerWIPType,
    XDaskWIPType,
    YDaskWIPType,
    GenericKFoldType
)

from ._get_kfold import _get_kfold
from ._fold_splitter import _fold_splitter
from ._parallelized_fit import _parallelized_fit
from ._parallelized_scorer import _parallelized_scorer
from ._parallelized_train_scorer import _parallelized_train_scorer

from model_selection.GSTCV._fit_shared._get_best_thresholds import \
    _get_best_thresholds

from model_selection.GSTCV._fit_shared._cv_results._cv_results_update import \
    _cv_results_update

from model_selection.GSTCV._fit_shared._cv_results. \
    _cv_results_rank_update import _cv_results_rank_update




def _core_fit(
    _X: XDaskWIPType,
    _y: YDaskWIPType,
    _estimator: ClassifierProtocol,
    _cv_results: CVResultsType,
    _cv: Union[int, GenericKFoldType],
    _error_score: Union[int, float, Literal['raise']],
    _verbose: int,
    _scorer: ScorerWIPType,
    _cache_cv: bool,
    _iid: bool,
    _return_train_score: bool,
    _PARAM_GRID_KEY: npt.NDArray[np.uint8],
    _THRESHOLD_DICT: dict[int, npt.NDArray[np.float64]],
    **params
    ) -> CVResultsType:

    """
    Perform all fit, scoring, and tabulation activities for every search
    performed in finding the hyperparameter values that maximize score
    (or minimize loss) for the given dataset (X) against the given target
    (y.)

    Returns all search results (times, scores, thresholds) in the
    cv_results dictionary.


    Parameters
    ----------
    _X:
        dask.array.core.Array[Union[int, float]] - the data to be fit by
        DaskGSTCV against the target.
    _y:
        dask.array.core.Array[Union[int, float]] - the target to train the data
        against.
    _estimator:
        Any classifier that fulfills the dask_ml API for classifiers,
        having fit, predict_proba, get_params, and set_params methods
        (the score method is not necessary, as DaskGSTCV never calls it.)
        This includes, but is not limited to, dask_ml, XGBoost, and LGBM
        classifiers.
    _cv_results:
        dict[str, np.ma.masked_array] - an unfilled cv_results dictionary,
        to store the times, scores, and thresholds found during the
        grid search process.
    _cv:
        Union[int, Iterable[tuple[Iterable, Iterable]] - as integer, the
        number of train/test splits to make using dask_ml KFold. As
        iterable (generator, vector-like), a sequence of tuples of
        vectors to be used as indices in making the train/test splits.
    _error_score:
        Union[int, float, Literal['raise']] - if a training fold excepts
        during fitting, the exception can be allowed to raise by passing
        the 'raise' literal. Otherwise, passing a number or number-like
        will cause the exception to be handled, allowing the grid search
        to proceed, and the given number carries through scoring
        tabulations in place of the missing scores.
    _verbose:
        int - a number from 0 to 10 that indicates the amount of
        information to display to the screen during the grid search
        process. 0 means no output, 10 means maximum output.
    _scorer:
        dict[str: Callable[[Iterable[int], Iterable[int]], float] -
        a dictionary with scorer name as keys and the scorer callables
        as values. The scorer callables are sklearn metrics, not
        make_scorer.
    _cache_cv:
        bool - Whether to extract each train/test subset at most once in
        each worker
        process, or every time that subset is needed. Caching the splits can
        speedup computation at the cost of increased memory usage per worker
        process.

        If True, worst case memory usage is ``(n_splits + 1) * (X.nbytes +
        y.nbytes)`` per worker. If False, worst case memory usage is
        ``(n_threads_per_worker + 1) * (X.nbytes + y.nbytes)`` per worker.
    _iid:
        bool - ndicates whether the data is believed to
        have random distribution of examples (True) or if the data is
        organized non-randomly in some way (False). If the data is not iid,
        dask KFold will cross chunk boundaries when reading the data in an
        attempt to randomize the data; this can be an expensive process.
        Otherwise, if the data is iid, dask KFold can handle the data as
        chunks which is much more efficient.
    _return_train_score:
        bool - If True, calculate scores for the train data in addition
        to the test data. There is a (perhaps appreciable) time and
        compute cost to this, as train sets are typically much bigger
        than test sets.

    _PARAM_GRID_KEY:
        npt.NDArray[np.uint8] - a vector of integers whose length is equal
        to the number of search permutations (also the number of rows in
        cv_results.) The integers indicate the index of the param grid
        in the param_grid list that provided the search points for the
        corresponding row in cv_results.
    _THRESHOLD_DICT:
        dict[int, npt.NDArray[np.float64]] - A dictionary whose values
        are the threshold vectors from each param grid in the param_grid
        list. Keyed by the index of the source param grid in the param
        grid list. The threshold vector is separated from its source
        param grid and put into this dictionary before building
        cv_results (if thresholds were left in param_grid and passed to
        the cv_results builder, they would be itemized out just like any
        other parameter; instead, each vector must be separated out and
        run in full for every search permutation.
    **params:
        **dict[str, any] - dictionary of kwarg: value pairs to be passed
        to the estimator's fit method.


    Return
    ------
    -
        _cv_results: dict[str: np.ma.masked_array] - dictionary populated
            with all the times, scores, thresholds, parameter values, and
            search grids for every permutation of grid search.

    """


    # rudimentarly stop-gap validation * * * * * *
    # this is just to force fast except on things that were exposed in
    # testing to not force except, or take a long time to except

    if not hasattr(_estimator, 'predict_proba'):
        raise AttributeError(f"_estimator must have predict_proba method")

    try:
        _n_splits = len(list(_cv))
        assert _n_splits  >= 2
    except:
        assert isinstance(_cv, int) and _cv >= 2, (f"_cv must be int >= 2 "
                       f"or a generator or an iterable with len >= 2")
        _n_splits = _cv

    try:
        float(_error_score)
        if isinstance(_error_score, bool):
            raise
    except:
        err_msg = f"_error_score must be a number or 'raise'"
        assert isinstance(_error_score, str), err_msg
        assert _error_score.lower() == 'raise', err_msg

    assert not isinstance(_verbose, bool) and 0 <= _verbose <= 10, \
        f"_verbose must be an int between 0 and 10 inclusive"

    assert isinstance(_cache_cv, bool), \
        f"_cache_cv must be bool"

    assert isinstance(_iid, bool), \
        f"_cache_cv must be bool"

    assert isinstance(_return_train_score, bool)

    assert isinstance(_scorer, dict)
    assert all(map(isinstance, _scorer, (str for _ in _scorer)))
    assert all(map(callable, _scorer.values()))

    assert all([int(_) == _ for _ in _PARAM_GRID_KEY])
    assert max(_PARAM_GRID_KEY) < len(_THRESHOLD_DICT)

    assert isinstance(_THRESHOLD_DICT, dict)
    assert all(map(isinstance, _THRESHOLD_DICT, (int for _ in _THRESHOLD_DICT)))

    # end validation * * * * * *

    CACHE_CV = None
    if isinstance(_cv, int):
        KFOLD = list(_get_kfold(_X, _n_splits, _iid, _verbose, _y=_y))
    else:  # _cv is an iterable
        KFOLD = list(_cv)

    if _cache_cv:
        CACHE_CV = []
        for (train_idxs, test_idxs) in KFOLD:
            CACHE_CV.append(_fold_splitter(train_idxs, test_idxs, _X, _y))
        KFOLD = None

    original_params = _estimator.get_params(deep=True)

    for trial_idx, _grid in enumerate(_cv_results['params']):

        if _verbose >= 3:
            print(f'\nparam grid {trial_idx + 1} of {len(_cv_results["params"])}: '
                  f'{_grid}')


        _THRESHOLDS = _THRESHOLD_DICT[_PARAM_GRID_KEY[trial_idx]]


        # reset the estimator to the first-seen params at every transition
        # to a new param grid, and then set the new params as called out
        # in cv_results_. in that way, the user can assume that params
        # not explicitly declared in a param grid are running at their
        # defaults (or whatever values they were hard-coded in when the
        # estimator was instantiated.)
        if trial_idx != 0:
            # at transition to the next param grid...
            if _PARAM_GRID_KEY[trial_idx] != _PARAM_GRID_KEY[trial_idx -1]:
                # ...put in the first-seen params...
                _estimator.set_params(**original_params)

        # ...then set the new params for the first search on the new grid
        _estimator.set_params(**_grid)

        ###################################################################
        # CORE GRID SEARCH ################################################
        # pizza 24_07_13 from old GSTCVDask.fit()
        # FIT ALL FOLDS ###################################################

















        # must use shallow params to construct estimator
        shallow_params = _estimator.get_params(deep=False)
        # must use deep params for pipeline to set GSCV params (depth
        # doesnt matter for an estimator.)
        deep_params = _estimator.get_params(deep=True)

        FIT_OUTPUT = list()
        if _cache_cv:
            # pizza! dont forget to put the fix for pipeline from SK in here!
            for f_idx, (_X_train, _, _y_train, _) in enumerate(CACHE_CV):

                FIT_OUTPUT.append(
                    _parallelized_fit(
                        f_idx,
                        _X_train,
                        _y_train,
                        type(_estimator)(**shallow_params).set_params(**deep_params),
                        _grid,
                        _error_score,
                        **params
                    )
                )

        elif not _cache_cv:
            for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD):

                # pizza! dont forget to put the fix for pipeline from SK in here!

                FIT_OUTPUT.append(
                    _parallelized_fit(
                        f_idx,
                        *_fold_splitter(train_idxs, test_idxs, _X, _y)[0::2],
                        type(_estimator)(**shallow_params).set_params(**deep_params),
                        _grid,
                        _error_score,
                        **params
                    )
                )

        # END FIT ALL FOLDS ###############################################

        # terminate if all folds excepted, display & compile fit times ** * ** *
        FOLD_FIT_TIMES_VECTOR = np.ma.empty(_n_splits, dtype=np.float64)
        FOLD_FIT_TIMES_VECTOR.mask = True
        num_failed_fits = 0


        # FIT_OUTPUT _estimator_, _fit_time, fit_excepted
        for idx, (_, _fit_time, _fit_excepted) in enumerate(FIT_OUTPUT):
            num_failed_fits += _fit_excepted

            FOLD_FIT_TIMES_VECTOR[idx] = np.ma.masked if _fit_excepted else _fit_time

            if _verbose >= 5:
                print(f'fold {idx + 1} train fit time = {_fit_time: ,.3g} s')

        if num_failed_fits == _n_splits:
            raise ValueError(f"all {_n_splits} folds failed during fit.")

        del idx, _, _fit_time, _fit_excepted, num_failed_fits
        # END terminate if all folds excepted, display & compile fit times ** *

        # SCORE ALL FOLDS & THRESHOLDS ########################################

        if _verbose >= 5:
            print(f'\nStart scoring test with different thresholds and scorers')

        test_predict_and_score_t0 = time.perf_counter()

        # TEST_SCORER_OUT IS:
        # TEST_THRESHOLD_x_SCORER__SCORE_LAYER,
        # TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER
        TEST_SCORER_OUT = list()
        if _cache_cv:

            for f_idx, (_, X_test, _, y_test) in enumerate(CACHE_CV):

                TEST_SCORER_OUT.append(
                    _parallelized_scorer(
                        # test only!
                        X_test,
                        y_test,
                        FIT_OUTPUT[f_idx],
                        f_idx,
                        _scorer,
                        _THRESHOLDS,
                        _error_score,
                        _verbose
                    )
                )

        elif not _cache_cv:

            for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD):

                TEST_SCORER_OUT.append(
                    _parallelized_scorer(
                        # test only!
                        *_fold_splitter(train_idxs, test_idxs, _X, _y)[1::2],
                        FIT_OUTPUT[f_idx],
                        f_idx,
                        _scorer,
                        _THRESHOLDS,
                        _error_score,
                        _verbose
                    )
                )

        test_predict_and_score_tf = time.perf_counter()
        tpast = test_predict_and_score_tf - test_predict_and_score_t0
        del test_predict_and_score_tf, test_predict_and_score_t0

        if _verbose >= 5:
            print(f'End scoring test with different thresholds and scorers')
            print(f'total test predict & score wall time = {tpast: ,.3g} s')

        del tpast

        # END SCORE ALL FOLDS & THRESHOLDS #################################

        # 3D-ify scores and times from parallel scorer ** * ** * ** *
        TSS = np.ma.masked_array(np.dstack([_[0] for _ in TEST_SCORER_OUT]))
        TSST = np.ma.masked_array(np.dstack([_[1] for _ in TEST_SCORER_OUT]))
        del TEST_SCORER_OUT

        TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX = TSS.transpose((2, 0, 1))
        TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX = TSST.transpose((2, 0, 1))
        del TSS, TSST
        # END 3D-ify scores and times from parallel scorer ** * ** * ** *

        # END CORE GRID SEARCH ############################################
        ###################################################################

        # NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ######

        # THIS CANNOT BE MELDED INTO ANYTHING ABOVE BECAUSE ALL FOLDS MUST
        # BE COMPLETED TO DO THIS
        TEST_BEST_THRESHOLD_IDXS_BY_SCORER = \
            _get_best_thresholds(
                TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                _THRESHOLDS,
        )
        # END NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ##

        # PICK THE COLUMNS FROM TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX
        # THAT MATCH RESPECTIVE BEST_THRESHOLDS_BY_SCORER
        # THIS NEEDS TO BE ma TO PRESERVE ANY MASKING DONE TO
        # TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX

        assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape == \
               (_n_splits, len(_THRESHOLDS), len(_scorer)), \
            "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"

        TEST_FOLD_x_SCORER__SCORE_MATRIX = \
            np.ma.empty((_n_splits, len(_scorer)))
        TEST_FOLD_x_SCORER__SCORE_MATRIX.mask = True

        for s_idx, t_idx in enumerate(TEST_BEST_THRESHOLD_IDXS_BY_SCORER):
            TEST_FOLD_x_SCORER__SCORE_MATRIX[:, s_idx] = \
                TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:, t_idx, s_idx]

        del TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX

        # SCORE TRAIN FOR THE BEST THRESHOLDS #############################

        # 24_02_21_13_57_00 ORIGINAL CONFIGURATION WAS TO DO BOTH TEST
        # SCORING AND TRAIN SCORING UNDER THE SAME FOLD LOOP FROM A
        # SINGLE FIT. BECAUSE FINAL THRESHOLD(S) CANT BE KNOWN YET, IT IS
        # IMPOSSIBLE TO SELECTIVELY GET BEST SCORES
        # FOR TRAIN @ THRESHOLD, SO ALL OF TRAIN'S SCORES MUST BE GENERATED.
        # AFTER FILLING TEST AND FINDING THE BEST THRESHOLDS,
        # THEN TRAIN SCORES CAN BE PICKED OUT. CALCULATING TRAIN SCORE TAKES
        # A LONG TIME FOR MANY THRESHOLDS, ESPECIALLY WITH
        # DASK. PERFORMANCE TESTS 24_02_21 INDICATE IT IS BETTER TO FIT
        # AND SCORE TEST ALONE, GET THE BEST THRESHOLD(S), THEN DO
        # ANOTHER LOOP FOR TRAIN WITH RETAINED COEFS FROM THE EARLIER FITS
        # TO ONLY GENERATE SCORES FOR THE SINGLE THRESHOLD(S).

        TRAIN_FOLD_x_SCORER__SCORE_MATRIX = \
            np.ma.zeros((_n_splits, len(_scorer)), dtype=np.float64)
        TRAIN_FOLD_x_SCORER__SCORE_MATRIX.mask = True

        if _return_train_score:

            _BEST_THRESHOLDS_BY_SCORER = _THRESHOLDS[
                TEST_BEST_THRESHOLD_IDXS_BY_SCORER
            ]

            # SCORE ALL FOLDS ###########################################

            if _verbose >= 5:
                print(f'\nStart scoring train with different scorers')

            train_predict_and_score_t0 = time.perf_counter()

            # TRAIN_SCORER_OUT is TRAIN_SCORER__SCORE_LAYER

            TRAIN_SCORER_OUT = []
            if _cache_cv:
                for f_idx, (X_train, _, y_train, _) in enumerate(CACHE_CV):
                    TRAIN_SCORER_OUT.append(
                        _parallelized_train_scorer(
                            # train only!
                            X_train,
                            y_train,
                            FIT_OUTPUT[f_idx],
                            f_idx,
                            _scorer,
                            _BEST_THRESHOLDS_BY_SCORER,
                            _error_score,
                            _verbose
                        )
                    )

            elif not _cache_cv:

                for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD):
                    TRAIN_SCORER_OUT.append(
                        _parallelized_train_scorer(
                            # train only!
                            *_fold_splitter(train_idxs, test_idxs, _X, _y)[0::2],
                            FIT_OUTPUT[f_idx],
                            f_idx,
                            _scorer,
                            _BEST_THRESHOLDS_BY_SCORER,
                            _error_score,
                            _verbose
                        )
                    )

            train_predict_and_score_tf = time.perf_counter()
            tpast = train_predict_and_score_tf - train_predict_and_score_t0
            del train_predict_and_score_tf, train_predict_and_score_t0

            if _verbose >= 5:
                print(f'End scoring train with different scorers')
                print(f'total train predict & score wall time = {tpast: ,.3g} s')

            del tpast

            # END SCORE ALL FOLDS #########################################

            del _BEST_THRESHOLDS_BY_SCORER

            TRAIN_FOLD_x_SCORER__SCORE_MATRIX = (
                np.ma.masked_array(np.vstack(TRAIN_SCORER_OUT))
            )

            del TRAIN_SCORER_OUT

        del FIT_OUTPUT
        # END SCORE TRAIN FOR THE BEST THRESHOLDS #########################

        # UPDATE cv_results_ WITH RESULTS #################################
        if _verbose >= 5:
            print(f'\nStart filling cv_results_')
            cv_t0 = time.perf_counter()

        # validate shape of holder objects before cv_results update ** * **
        assert FOLD_FIT_TIMES_VECTOR.shape == (_n_splits,), \
            "FOLD_FIT_TIMES_VECTOR is misshapen"
        assert TEST_FOLD_x_SCORER__SCORE_MATRIX.shape == \
               (_n_splits, len(_scorer)), \
            f"TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"
        assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape == \
               (_n_splits, len(_THRESHOLDS), len(_scorer)), \
            "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX is misshapen"
        assert TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape == \
               (_n_splits, len(_scorer)), \
            "TRAIN_FOLD_x_SCORER__SCORE_MATRIX is misshapen"
        # END validate shape of holder objects before cv_results update **

        _cv_results = _cv_results_update(
            trial_idx,
            _THRESHOLDS,
            FOLD_FIT_TIMES_VECTOR,
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX,
            TEST_BEST_THRESHOLD_IDXS_BY_SCORER,
            TEST_FOLD_x_SCORER__SCORE_MATRIX,
            TRAIN_FOLD_x_SCORER__SCORE_MATRIX,
            _scorer,
            _cv_results,
            _return_train_score
        )

        if _verbose >= 5:
            cv_tf = time.perf_counter()
            print(f'End filling cv_results_ = {cv_tf - cv_t0: ,.3g} s')
            del cv_t0, cv_tf

            del TEST_FOLD_x_SCORER__SCORE_MATRIX, TRAIN_FOLD_x_SCORER__SCORE_MATRIX
            del FOLD_FIT_TIMES_VECTOR, TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX



    # FINISH RANK COLUMNS HERE #####################################
    # ONLY DO TEST COLUMNS, DONT DO TRAIN RANK
    _cv_results_ = _cv_results_rank_update(
        _scorer,
        _cv_results
    )
    # END FINISH RANK COLUMNS HERE #################################

    # 24_07_16, when testing against dask GSCV, this module is altering
    # the params of the in-scope estimator (estimator used inside this
    # module) and is altering the estimator that is out-of-scope (the
    # one that is passed to this module) even though this scope's estimator
    # is not being returned. to prevent any future problems, set estimator's
    # params back to the way they started
    _estimator.set_params(**original_params)

    del original_params


    return _cv_results








