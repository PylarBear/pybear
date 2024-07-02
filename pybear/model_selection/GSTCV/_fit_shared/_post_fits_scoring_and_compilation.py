# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


def _post_fits_scoring_and_compilation(
    f_idx,
    X_test,
    y_test,
    FIT_OUTPUT,

    ) -> :


    # COMPILE INFO FROM FIT ###########################################

    for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD):

        X_test, y_test = _fold_splitter(train_idxs, test_idxs, X, y)[1::2]

        _estimator_, _fit_time, fit_excepted = FIT_OUTPUT[f_idx]

        TEST_FOLD_FIT_TIME_VECTOR[f_idx] = _fit_time

        if self.verbose >= 5:
            print(f'fold {f_idx + 1} train fit time = {_fit_time: ,.3g} s')

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

        FITTED_ESTIMATOR_COEFS[f_idx] = _estimator_.coef_

        # END COMPILE INFO FROM FIT #######################################

        # GET SCORE FOR ALL THRESHOLDS ####################################

        test_predict_and_score_t0 = time.perf_counter()
        if self.verbose >= 5:
            print(f'Start scoring with different thresholds and scorers')

        for thresh_idx, _threshold in enumerate(THRESHOLDS):

            try:
                y_test = y_test.compute(scheduler=self.scheduler)
            except:
                pass

            _y_test_pred = (_predict_proba >= _threshold).astype(np.uint8)

            for s_idx, scorer_key in enumerate(self.scorer_):
                test_t0_score = time.perf_counter()
                __ = self.scorer_[scorer_key](y_test, _y_test_pred, **scorer_params)
                test_tf_score = time.perf_counter()
                TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[f_idx, thresh_idx, s_idx] = __
                TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, thresh_idx, :] = test_tf_score - test_t0_score
            del __

        if self.verbose >= 5:
            print(f'End scoring with different thresholds and scorers')

        test_predict_and_score_tf = time.perf_counter()

        if self.verbose >= 5:
            FOLD_TIMES = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :]
            print(f'fold {f_idx + 1} total tests score time = {FOLD_TIMES.sum(): ,.3g} s')
            print(f'fold {f_idx + 1} avg tests score time = {FOLD_TIMES.mean(): ,.3g} s')
            del FOLD_TIMES
            print(f'fold {f_idx + 1} total tests predict & score wall time = {test_predict_and_score_tf - test_predict_and_score_t0: ,.3g} s')

    del FIT_OUTPUT, X_test, y_test, _predict_proba
    del test_predict_and_score_t0, test_predict_and_score_tf
    # END GET SCORE FOR ALL THRESHOLDS ################################

    # END CORE GRID SEARCH ############################################
    ###################################################################