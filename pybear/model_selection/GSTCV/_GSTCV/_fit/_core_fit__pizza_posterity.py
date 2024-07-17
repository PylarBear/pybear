# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# pizza pizza!
# 24_07_09_10_07_00 this was the old code that was under fit() in GSTCV
# before being replaced by _core_fit(). Keep this for posterity just in
# case anything was missed (like selfs).





for trial_idx, _grid in enumerate(self.cv_results_['params']):

    # THRESHOLDS MUST BE IN wip_param_grid(s). REMOVE IT AND SET IT
    # ASIDE, BECAUSE IT CANT BE PASSED TO estimator
    # ONLY DO THIS FIRST TIME ACCESSING A NEW wip_param_grid

    if self.verbose >= 3:
        print(
            f'\nparam grid {trial_idx + 1} of {len(self.cv_results_["params"])}: {_grid}')

    THRESHOLDS = THRESHOLD_DICT[PARAM_GRID_KEY[trial_idx]]

    self._estimator.set_params(**_grid)

    ###################################################################
    # CORE GRID SEARCH ################################################

    if isinstance(self.cv, int):
        KFOLD = list(_get_kfold(X, y, self.n_splits_, self.verbose))
    else:
        KFOLD = self.cv

    # FIT ALL FOLDS ###################################################

    with joblib.parallel_config(prefer='processes', n_jobs=self.n_jobs):
        FIT_OUTPUT = joblib.Parallel(return_as='list')(
            joblib.delayed(_parallelized_fit)(
                f_idx,
                *_fold_splitter(train_idxs, test_idxs, X, y)[::2],
                # train only!
                train_idxs,
                test_idxs,
                self._estimator,
                _grid,
                self.error_score,
                **params
            ) for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD)
        )

    # END FIT ALL FOLDS ###############################################

    # terminate if all folds excepted ** * ** * ** * ** * ** * **
    num_failed_fits = 0
    for _trial, out_tuple in enumerate(FIT_OUTPUT):
        num_failed_fits += out_tuple[-1]

        if self.verbose >= 5:
            print(f'fold {_trial + 1} train fit time = {out_tuple[1]: ,.3g} s')

    num_failed_fits += 1

    if num_failed_fits == self.n_splits_:
        raise ValueError(f"all {self.n_splits_} folds failed during fit.")

    del num_failed_fits
    # END terminate if all folds excepted ** * ** * ** * ** * **

    # compile fit times ** * ** * ** * ** * ** * ** * ** * ** * **
    FOLD_FIT_TIMES_VECTOR = np.ma.empty(self.n_splits_, dtype=np.float64)
    # FIT_OUTPUT _estimator_, _fit_time, fit_excepted
    for idx, (_, _fit_time, _fit_excepted) in enumerate(FIT_OUTPUT):

        if _fit_excepted:
            FOLD_FIT_TIMES_VECTOR[idx] = np.ma.masked
        else:
            FOLD_FIT_TIMES_VECTOR[idx] = _fit_time

    del idx, _, _fit_time, _fit_excepted
    # END compile fit times ** * ** * ** * ** * ** * ** * ** * **

    # SCORE ALL FOLDS & THRESHOLDS #####################################

    if self.verbose >= 5:
        print(f'Start scoring with different thresholds and scorers')

    test_predict_and_score_t0 = time.perf_counter()

    SCORER_OUT = joblib.Parallel(return_as='list')(
        joblib.delayed(_parallelized_scorer)(
            *_fold_splitter(train_idxs, test_idxs, X, y)[1::2],  # test only!
            FIT_OUTPUT[f_idx],
            f_idx,
            self.scorer_,
            self.n_splits_,
            THRESHOLDS,
            self.error_score,
            self.verbose
        ) for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD)
    )

    test_predict_and_score_tf = time.perf_counter()
    tpast = test_predict_and_score_tf - test_predict_and_score_t0
    del test_predict_and_score_tf, test_predict_and_score_t0

    # END SCORE ALL FOLDS & THRESHOLDS #################################

    # SCORER_OUT IS:
    # TEST_THRESHOLD_x_SCORER__SCORE_LAYER,
    # TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER

    # 3D-ify scores and times from parallel scorer ** * ** * ** *
    TSS = np.ma.masked_array(np.dstack([_[0] for _ in SCORER_OUT]))
    TSST = np.ma.masked_array(np.dstack([_[1] for _ in SCORER_OUT]))
    del SCORER_OUT

    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX = TSS.transpose((2, 0, 1))
    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX = TSST.transpose(
        (2, 0, 1))
    del TSS, TSST
    # 3D-ify scores and times from parallel scorer ** * ** * ** *

    if self.verbose >= 5:
        print(f'End scoring with different thresholds and scorers')
        for f_idx in range(self.n_splits_):
            _ = f_idx + 1
            __ = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :]
            print(f'fold {_} total tests score time = {__.sum(): ,.3g} s')
            print(f'fold {_} avg tests score time = {__.mean(): ,.3g} s')
            del _, __

        print(f'total test predict & score wall time = {tpast: ,.3g} s')

    del tpast

    # END CORE GRID SEARCH ############################################
    ###################################################################

    # NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ######

    # THIS CANNOT BE MELDED INTO ANYTHING ABOVE BECAUSE ALL FOLDS MUST
    # BE COMPLETED TO DO THIS
    TEST_BEST_THRESHOLD_IDXS_BY_SCORER = \
        _get_best_thresholds(
            trial_idx,
            list(self.scorer_.keys()),
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
            THRESHOLDS
        )
    # END NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ##

    # PICK THE COLUMNS FROM TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX
    # THAT MATCH RESPECTIVE TEST_BEST_THRESHOLD_IDXS_BY_SCORER
    # THIS NEEDS TO BE ma TO PRESERVE ANY MASKING DONE TO
    # TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX

    assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape == \
           (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
        "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"

    TEST_FOLD_x_SCORER__SCORE_MATRIX = \
        np.ma.empty((self.n_splits_, len(self.scorer_)))

    for s_idx, t_idx in enumerate(TEST_BEST_THRESHOLD_IDXS_BY_SCORER):
        TEST_FOLD_x_SCORER__SCORE_MATRIX[:, s_idx] = \
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:, t_idx, s_idx]

    del TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX

    # SCORE TRAIN FOR THE BEST THRESHOLDS ######################

    # 24_02_21_13_57_00 ORIGINAL CONFIGURATION WAS TO DO BOTH TEST
    # SCORING AND TRAIN SCORING UNDER THE SAME FOLD LOOP FROM A
    # SINGLE FIT. BECAUSE FINAL THRESHOLD(S) CANT BE KNOWN YET,
    # IT IS IMPOSSIBLE TO SELECTIVELY GET BEST SCORES FOR TRAIN @
    # THRESHOLD, SO ALL OF TRAIN'S SCORES MUST BE GENERATED. AFTER
    # FILLING TEST AND FINDING THE BEST THRESHOLDS, THEN TRAIN
    # SCORES CAN BE PICKED OUT. CALCULATING TRAIN SCORE TAKES A
    # LONG TIME FOR MANY THRESHOLDS, ESPECIALLY WITH DASK.
    # PERFORMANCE TESTS 24_02_21 INDICATE IT IS BETTER TO FIT AND
    # SCORE TEST ALONE, GET THE BEST THRESHOLD(S), THEN DO ANOTHER
    # LOOP FOR TRAIN WITH RETAINED ESTIMATORS FROM THE EARLIER FITS
    # TO ONLY GENERATE SCORES FOR THE SINGLE THRESHOLDS.

    TRAIN_FOLD_x_SCORER__SCORE_MATRIX = \
        np.ma.zeros((self.n_splits_, len(self.scorer_)), dtype=np.float64)

    if self.return_train_score:

        _BEST_THRESHOLDS_BY_SCORER = THRESHOLDS[
            TEST_BEST_THRESHOLD_IDXS_BY_SCORER
        ]

        train_predict_and_score_t0 = time.perf_counter()

        # TRAIN_SCORER_OUT is TRAIN_SCORER__SCORE_LAYER

        TRAIN_SCORER_OUT = joblib.Parallel(return_as='list')(
            joblib.delayed(_parallelized_train_scorer)(
                # train only!
                *_fold_splitter(train_idxs, test_idxs, X, y)[0::2],
                FIT_OUTPUT[f_idx],
                f_idx,
                self.scorer_,
                _BEST_THRESHOLDS_BY_SCORER,
                self.error_score,
                self.verbose
            ) for f_idx, (train_idxs, test_idxs) in
            enumerate(KFOLD)
        )

        train_predict_and_score_tf = time.perf_counter()
        tpast = train_predict_and_score_tf - train_predict_and_score_t0
        del train_predict_and_score_tf, train_predict_and_score_t0

        if self.verbose >= 5:
            print(f'total train predict & score wall time = {tpast: ,.3g} s')

        del tpast

        del _BEST_THRESHOLDS_BY_SCORER

        TRAIN_FOLD_x_SCORER__SCORE_MATRIX = np.vstack(TRAIN_SCORER_OUT)

        del TRAIN_SCORER_OUT

    del FIT_OUTPUT
    # END SCORE TRAIN FOR THE BEST THRESHOLDS #########################

    # UPDATE cv_results_ WITH RESULTS #################################
    if self.verbose >= 5:
        print(f'\nStart filling cv_results_')
        cv_t0 = time.perf_counter()

    # validate shape of holder objects before cv_results update ** * **
    assert FOLD_FIT_TIMES_VECTOR.shape == (self.n_splits_,), \
        "FOLD_FIT_TIMES_VECTOR is misshapen"
    assert TEST_FOLD_x_SCORER__SCORE_MATRIX.shape == \
           (self.n_splits_, len(self.scorer_)), \
        f"TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"
    assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape == \
           (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
        "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX is misshapen"
    assert TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape == \
           (self.n_splits_, len(self.scorer_)), \
        "TRAIN_FOLD_x_SCORER__SCORE_MATRIX is misshapen"
    # END validate shape of holder objects before cv_results update **

    self.cv_results_ = _cv_results_update(
        trial_idx,
        THRESHOLDS,
        FOLD_FIT_TIMES_VECTOR,
        TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX,
        TEST_BEST_THRESHOLD_IDXS_BY_SCORER,
        TEST_FOLD_x_SCORER__SCORE_MATRIX,
        TRAIN_FOLD_x_SCORER__SCORE_MATRIX,
        self.scorer_,
        self.cv_results_,
        self.return_train_score
    )

    if self.verbose >= 5:
        cv_tf = time.perf_counter()
        print(f'End filling cv_results_ = {cv_tf - cv_t0: ,.3g} s')
        del cv_t0, cv_tf

    del TEST_FOLD_x_SCORER__SCORE_MATRIX, TRAIN_FOLD_x_SCORER__SCORE_MATRIX
    del FOLD_FIT_TIMES_VECTOR, TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX

# FINISH RANK COLUMNS HERE #####################################
# ONLY DO TEST COLUMNS, DONT DO TRAIN RANK
self.cv_results_ = _cv_results_rank_update(
    self.scorer_,
    self.cv_results_
)
# END FINISH RANK COLUMNS HERE #################################





