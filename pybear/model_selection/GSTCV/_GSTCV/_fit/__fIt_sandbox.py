# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


for trial_idx, _grid in enumerate(self.cv_results_['params']):

    # THRESHOLDS MUST BE IN wip_param_grid(s). REMOVE IT AND SET IT ASIDE, BECAUSE IT CANT BE PASSED TO estimator
    # ONLY DO THIS FIRST TIME ACCESSING A NEW wip_param_grid

    if self.verbose >= 3:
        print(f'\nparam grid {trial_idx + 1} of {param_permutations}: {_grid}')

    pgk = PARAM_GRID_KEY[trial_idx]
    if trial_idx == 0 or (pgk != PARAM_GRID_KEY[trial_idx - 1]):
        THRESHOLDS = THRESHOLD_DICT[trial_idx]
    del pgk

    self._estimator.set_params(**_grid)

    FITTED_ESTIMATOR_COEFS = np.empty(self.n_splits_, dtype=object)  # KEEP THE COEFS IN CASE THEY ARE NEEDED FOR TRAIN SCORING
    TEST_FOLD_FIT_TIME_VECTOR = np.empty(self.n_splits_, dtype=np.float64)
    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX = np.ma.empty((self.n_splits_, len(THRESHOLDS), len(self.scorer_)), dtype=np.float64)
    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX = np.ma.empty((self.n_splits_, len(THRESHOLDS), len(self.scorer_)), dtype=np.float64)
    TRAIN_FOLD_x_SCORER__SCORE_MATRIX = np.ma.empty((self.n_splits_, len(self.scorer_)))

    assert FITTED_ESTIMATOR_COEFS.shape == (self.n_splits_,), "FITTED_ESTIMATOR_COEFS is misshapen"
    assert TEST_FOLD_FIT_TIME_VECTOR.shape == (self.n_splits_,), "TEST_FOLD_FIT_TIME_VECTOR is misshapen"
    assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape == (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
        "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"
    assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape == (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
        "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX is misshapen"
    assert TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape == (self.n_splits_, len(self.scorer_)), "TRAIN_FOLD_x_SCORER__SCORE_MATRIX is misshapen"

    ###################################################################
    # CORE GRID SEARCH ################################################

    # FIT ALL FOLDS ###################################################

    @joblib.wrap_non_picklable_objects
    def _parallelized_fit(
            f_idx: int,
            X_train: Union[XSKWIPType, XDaskWIPType],
            y_train: Union[YSKWIPType, YDaskWIPType],
            train_idxs: SKKFoldType,
            test_idxs: SKKFoldType,
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
                raise ValueError(
                    f"estimator excepted during fitting on {_grid}, "
                    f"cv fold index {f_idx} --- {f}")
            else:
                fit_excepted = True
                warnings.warn(
                    f'\033[93mfit excepted on {_grid}, cv fold index {f_idx}\033[0m'
                )

        _fit_time = time.perf_counter() - t0_fit

        del t0_fit

        # train_idxs, test_idxs must be available when unpacking the joblib list
        return _estimator_, _fit_time, fit_excepted, train_idxs, test_idxs


    ARGS_FOR_PARALLEL_FIT = (self._estimator, _grid, self.error_score)

    with joblib.parallel_config(prefer='processes', n_jobs=self.n_jobs):
        FIT_OUTPUT = joblib.Parallel(return_as='list')(
            joblib.delayed(_parallelized_fit)(
                f_idx,
                *_fold_splitter(train_idxs, test_idxs, X, y)[::2],
                train_idxs,
                test_idxs,
                *ARGS_FOR_PARALLEL_FIT,
                **fit_params) for f_idx, (train_idxs, test_idxs) in enumerate(_get_kfold(X, y, self.n_splits_, self.verbose))
        )

    del ARGS_FOR_PARALLEL_FIT
    # END FIT ALL FOLDS ###############################################

    # COMPILE INFO FROM FIT ###########################################

    ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS = np.ma.empty((self.n_splits_, 2, X.shape[0] // self.n_splits_ + 1), dtype=np.float64)
    LEN_Y_TEST = list()
    FIT_EXCEPTED = list()

    for f_idx, (_estimator_, _fit_time, fit_excepted, train_idxs, test_idxs) in enumerate(FIT_OUTPUT):

        X_test, y_test = _fold_splitter(np.arange(len(X) - len(test_idxs.ravel())), test_idxs, X, y)[1::2]

        LEN_Y_TEST.append(len(test_idxs.ravel()))
        # SET y_test (ALWAYS, WHETHER OR NOT fit() EXCEPTED)
        ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS[f_idx, 1, :LEN_Y_TEST[f_idx]] = y_test.ravel()

        TEST_FOLD_FIT_TIME_VECTOR[f_idx] = _fit_time

        if self.verbose >= 5:
            print(f'fold {f_idx + 1} train fit time = {_fit_time: ,.3g} s')

        FIT_EXCEPTED.append(fit_excepted)

        # IF A FOLD EXCEPTED DURING FIT, ALL THE THRESHOLDS AND SCORERS
        # IN THAT FOLD LAYER GET SET TO error_score.
        # SCORE TIME CANT BE TAKEN SINCE SCORING WONT BE DONE SO ALSO MASK THAT
        if fit_excepted:
            FITTED_ESTIMATOR_COEFS[f_idx] = None
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[f_idx, :, :] = self.error_score
            if self.error_score is np.nan:
                TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[f_idx, :, :] = np.ma.masked

            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :] = np.nan
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :] = np.ma.masked

            # SET predict_proba
            ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS[f_idx, 0, :] = np.nan
            ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS[f_idx, 0, :] = np.ma.masked

            continue  # GO TO NEXT FOLD

        pp_time = time.perf_counter()
        _predict_proba = _estimator_.predict_proba(X_test)[:, -1].ravel()
        try:
            _predict_proba = _predict_proba.compute(scheduler=self.scheduler)
        except:
            pass

        if self.verbose >= 5:
            print(
                f'fold {f_idx + 1} tests predict_proba time = {time.perf_counter() - pp_time: ,.3g} s')
        del pp_time

        # SET predict_proba
        ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS[f_idx, 0, :LEN_Y_TEST[f_idx]] = _predict_proba
        del _predict_proba

        FITTED_ESTIMATOR_COEFS[f_idx] = _estimator_.coef_

    del FIT_OUTPUT, X_test, y_test
    # END COMPILE INFO FROM FIT #######################################

    # GET SCORE FOR ALL THRESHOLDS ####################################
    for f_idx, (_predict_proba, y_test) in enumerate(ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS):

        if FIT_EXCEPTED[f_idx]:
            continue

        _predict_proba = _predict_proba[:LEN_Y_TEST[f_idx]]
        y_test = y_test[:LEN_Y_TEST[f_idx]]

        test_predict_and_score_t0 = time.perf_counter()
        if self.verbose >= 5: print(
            f'Start scoring with different thresholds and scorers')


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




        joblib_kwargs = {
            'prefer': 'processes',
            'n_jobs': self.n_jobs
        }
        with joblib.parallel_config(**joblib_kwargs):
            JOBLIB_TEST_SCORE_OUTPUT = \
                joblib.Parallel(return_as='list')(
                    joblib.delayed(_threshold_scorer_sweeper)(
                        threshold,
                        y_test,
                        _predict_proba,
                        self.scorer_,
                        **scorer_params
                    ) for threshold in THRESHOLDS
                )

        del joblib_kwargs

        if self.verbose >= 5: print(
            f'End scoring with different thresholds and scorers')
        test_predict_and_score_tf = time.perf_counter()

        for thresh_idx, RESULT_SET in enumerate(JOBLIB_TEST_SCORE_OUTPUT):
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[f_idx, thresh_idx, :] = RESULT_SET[0]
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, thresh_idx, :] = RESULT_SET[1]

        del JOBLIB_TEST_SCORE_OUTPUT

        if self.verbose >= 5:
            FOLD_TIMES = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :]
            print(f'fold {f_idx + 1} total tests score time = {FOLD_TIMES.sum(): ,.3g} s')
            print(f'fold {f_idx + 1} avg tests score time = {FOLD_TIMES.mean(): ,.3g} s')
            del FOLD_TIMES
            print(f'fold {f_idx + 1} total tests predict & score wall time = {test_predict_and_score_tf - test_predict_and_score_t0: ,.3g} s')

    del ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS, LEN_Y_TEST, FIT_EXCEPTED, y_test, _predict_proba
    del test_predict_and_score_t0, test_predict_and_score_tf
    # END GET SCORE FOR ALL THRESHOLDS ################################

    # END CORE GRID SEARCH ############################################
    ###################################################################

    # NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ######

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
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:, :,
                s_idx].mean(axis=0)

            _SCORER_THRESH_MEANS = _SCORER_THRESH_MEANS.ravel()

            assert len(_SCORER_THRESH_MEANS) == len(_THRESHOLDS), \
                f"len(_SCORER_THRESH_MEANS) != len(_THRESHOLDS)"

            # IF MULTIPLE THRESHOLDS HAVE BEST SCORE, USE THE ONE CLOSEST TO 0.50
            # FIND CLOSEST TO 0.50 USING (THRESH - 0.50)**2
            BEST_SCORE_IDX_MASK = (
                        _SCORER_THRESH_MEANS == _SCORER_THRESH_MEANS.max())
            del _SCORER_THRESH_MEANS

            MASKED_LSQ = (
                        1 - np.power(_THRESHOLDS - 0.50, 2, dtype=np.float64))
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


    TEST_BEST_THRESHOLD_IDXS_BY_SCORER = _get_best_thresholds(
        trial_idx,
        list(self.scorer_.keys()),
        TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
        THRESHOLDS,
        self.cv_results_
    )
    # END NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ##

    # PICK THE COLUMNS FROM TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX THAT MATCH RESPECTIVE BEST_THRESHOLDS_BY_SCORER
    # THIS NEEDS TO BE ma TO PRESERVE ANY MASKING DONE TO TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX
    TEST_FOLD_x_SCORER__SCORE_MATRIX = np.ma.empty(
        (self.n_splits_, len(self.scorer_)))

    for scorer_idx, threshold_idx in zip(range(len(self.scorer_)),
                                         TEST_BEST_THRESHOLD_IDXS_BY_SCORER):
        TEST_FOLD_x_SCORER__SCORE_MATRIX[:,
        scorer_idx] = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:,
                      threshold_idx, scorer_idx]

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

    if self.return_train_score:

        # SINCE COPIES OF self._estimator WERE SENT TO parallel_fit, self._estimator HAS NEVER BEEN FIT AND
        # THEREFORE _estimator.predict_proba HAS NOT BEEN EXPOSED. EXPOSE IT WITH A DUMMY FIT.
        try:
            DUM_DATA = X.loc[:2,
                       :]  # PASS WITH COLUMN NAMES IF WAS A DF TO GET feature_names_in_
        except:
            DUM_DATA = X[:2, :]
        self._estimator.fit(DUM_DATA, [0, 1])
        del DUM_DATA

        for f_idx, (train_idxs, test_idxs) in enumerate(_get_kfold(X, y, self.n_splits_, self.verbose)):
            # IF FITTED_ESTIMATOR_COEFS[f_idx] IS None, IT IS BECAUSE fit() EXCEPTED ON THAT FOLD.
            # PUT IN error_score IF error_score!=np.nan, elif is np.nan ALSO MASK THIS FOLD IN
            # TRAIN_FOLD_x_SCORER__SCORE_MATRIX, AND SKIP TO NEXT FOLD
            if FITTED_ESTIMATOR_COEFS[f_idx] is None:
                TRAIN_FOLD_x_SCORER__SCORE_MATRIX[f_idx, :] = self.error_score
                if self.error_score is np.nan:
                    TRAIN_FOLD_x_SCORER__SCORE_MATRIX[f_idx, :] = np.ma.masked

                continue

            X_train, y_train = _fold_splitter(train_idxs, test_idxs, X, y)[
                               0::2]

            assert X_train.shape[0] == y_train.shape[
                0], "X_train.shape[0] != y_train.shape[0]"

            # PUT THE RETAINED COEFS FROM THE fit() EARLIER BACK INTO estimator
            self._estimator.coef_ = FITTED_ESTIMATOR_COEFS[f_idx]
            _predict_proba = self._estimator.predict_proba(X_train)[:, -1]
            try:
                _predict_proba = _predict_proba.compute(
                    scheduler=self.scheduler)
            except:
                pass

            assert len(_predict_proba) == X_train.shape[0], \
                "len(_predict_proba) != X_train.shape[0]"

            train_predict_and_score_t0 = time.perf_counter()
            train_pred_time, train_score_time = 0, 0
            for s_idx, scorer_key in enumerate(self.scorer_):
                train_pred_t0 = time.perf_counter()
                y_train_pred = (_predict_proba >= THRESHOLDS[
                    TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx]])
                train_pred_time += (time.perf_counter() - train_pred_t0)
                assert len(y_train_pred) == X_train.shape[0], \
                    "len(y_train_pred) != X_train.shape[0]"
                train_score_t0 = time.perf_counter()
                __ = self.scorer_[scorer_key](y_train, y_train_pred)
                train_score_time += (time.perf_counter() - train_score_t0)
                assert (__ >= 0 and __ <= 1), \
                    f"{scorer_key} score is not 0 <= score <= 1"
                TRAIN_FOLD_x_SCORER__SCORE_MATRIX[f_idx, s_idx] = __

            if self.verbose >= 5:
                print(
                    f'fold {f_idx + 1} total train predict time = {train_pred_time: ,.3g} s')
                print(
                    f'fold {f_idx + 1} total train score time = {train_score_time: ,.3g} s')
                print(
                    f'fold {f_idx + 1} avg train score time = {train_score_time / len(self.scorer_): ,.3g} s')
                print(
                    f'fold {f_idx + 1} total train predict and score time = {time.perf_counter() - train_predict_and_score_t0: ,.3g} s')

        try:
            del X_train, y_train, _predict_proba, y_train_pred, __, train_predict_and_score_t0, train_pred_time, \
                train_score_time, train_pred_t0, train_pred_time, train_score_t0, train_score_time
        except:
            pass

    del FITTED_ESTIMATOR_COEFS, TEST_BEST_THRESHOLD_IDXS_BY_SCORER
    # END SCORE TRAIN FOR THE BEST THRESHOLDS #########################






































