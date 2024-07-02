# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
import time, sys, warnings
from typing import Union

import numpy as np
import scipy.stats as ss
import joblib
from dask import compute, visualize
from dask.distributed import Client

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    ScorerInputType,
    ScorerWIPType,
    ClassifierProtocol,
    ParamGridType,
    SchedulerType
)


from ._validation._is_dask_estimator import _is_dask_estimator

from pybear.base import is_classifier as pybear_is_classifier

from model_selection.GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from model_selection.GSTCV._cv_results_builder._cv_results_builder import \
    _cv_results_builder

from ._handle_X_y._handle_X_y_dask import _handle_X_y_dask


from model_selection.GSTCV._GSTCVDask._fit._get_kfold import _get_kfold
from model_selection.GSTCV._GSTCVDask._fit import _fold_splitter
from model_selection.GSTCV._GSTCV._fit import _parallelized_fit

from model_selection.GSTCV._fit_shared import (
    _threshold_scorer_sweeper,
    _get_best_thresholds
)
from model_selection.GSTCV._fit_shared._cv_results_update import \
    _cv_results_score_updater


class GSTCVDask(_GSTCVMixin):

    """

    --- Classifer must have predict_proba method. If does not have predict_proba,
    try to wrap with CalibratedClassifierCV.



    Notes
    -----
    The parameters selected are those that maximize the score of the left
    out data, unless an explicit score is passed in which case it is
    used instead.  ???




    Examples
    --------
    >>> from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
    >>> from sklearn import svm, datasets

    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
    >>> svc = svm.SVC()

    >>> clf = dask_GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearchCV(cache_cv=..., cv=..., error_score=...,
        estimator=SVC(C=..., cache_size=..., class_weight=..., coef0=...,
                      decision_function_shape=..., degree=..., gamma=...,
                      kernel=..., max_iter=-1, probability=False,
                      random_state=..., shrinking=..., tol=...,
                      verbose=...),
        iid=..., n_jobs=..., param_grid=..., refit=..., return_train_score=...,
        scheduler=..., scoring=...)

    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]
    """

    def __init__(self,
        estimator: ClassifierProtocol,
        param_grid: ParamGridType,
        # thresholds can be a single number or list-type passed in
        # param_grid or applied universally via thresholds kwarg
        *,

        thresholds: Union[np.ndarray[int,float],list[int, float],int,float]=None,  # pizza
        scoring: ScorerInputType=None,
        iid: bool = True,
        refit: Union[callable, bool, str, list[str], None] = True,
        cv: Union[int, None] = None,
        error_score: Union[str, int, float] = np.nan,  # pizza
        return_train_score: bool = False,
        scheduler: SchedulerType = None,
        n_jobs: Union[int,None]=1,
        cache_cv:bool=True
        ):


        self._estimator = estimator
        self.param_grid = param_grid
        self.thresholds = thresholds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.iid = iid
        self.scheduler = scheduler
        self.cache_cv = cache_cv

    # END init ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def _handle_X_y(self, X, y=None):
        return _handle_X_y_dask(X, y=y)

    ####################################################################
    # SKLEARN / DASK GridSearchCV Methods ##############################

    def fit(self, X: XInputType, y: YInputType=None, **params):

        """
        Analog to dask/sklearn GridSearchCV fit() method. Run fit with
        all sets of parameters.
        Pizza add words.

        Parameters
        ----------
        # pizza can y be [] and [[]]
        X: Iterable[Iterable[Union[int, float]]] - training data
        y: Union[Iterable[Iterable[Union[int,float]]], Iterable[Union[int,float]]] -
            target for training data
        groups: Group labels for the samples used while splitting the dataset
            into train/tests set
        **params: ???

        Return
        ------
        -
            Instance of fitted estimator.


        """


        _X, _y, _feature_names_in, _n_features_in = self._handle_X_y(X, y)


        # DONT unique().compute() HERE, JUST RETAIN THE VECTOR & ONLY DO THE PROCESSING IF classes_ IS CALLED
        self._classes_ = y

        # THIS IS A HOLDER THAT IS FILLED ONE TIME WHEN THE unique().compute() IS DONE ON self._classes_
        self._classes = None


        # BUILD FROM **params
        kfold_params = {}
        fit_params = {}
        scorer_params = {}
        refit_params = {}


        self.validate_and_reset()

        # BEFORE RUNNING cv_results_builder, THE THRESHOLDS MUST BE REMOVED FROM EACH PARAM GRID IN wip_param_grids
        # BUT THEY NEED TO BE RETAINED FOR CORE GRID SEARCH.
        THRESHOLD_DICT = {i:self.wip_param_grid[i].pop('thresholds') for i in range(len(self.wip_param_grid))}

        self.cv_results_, PARAM_GRID_KEY = \
            _cv_results_builder(self.wip_param_grid, self.n_splits_, self.scorer_, self.return_train_score)

        param_permutations = len(self.cv_results_['params'])


        # IF refit IS A FUNCTION, FILL cv_results WITH DUMMY DATA, AND SEE
        # IF THE FUNCTION RETURNS AN INTEGER IN RANGE OF cv_results_ BEFORE
        # RUNNING THE ENTIRETY OF GRIDSEARCH WHICH COULD BE HOURS OR DAYS
        # JUST TO HAVE THE WHOLE THING CRASH BECAUSE OF A BAD refit FUNCTION
        if callable(self.wip_refit):
            DUMMY_CV_RESULTS = deepcopy(self.cv_results_)
            for column in DUMMY_CV_RESULTS:
                if column[:5]=='param' or column[:4]=='rank':
                    continue
                else:
                    DUMMY_CV_RESULTS[column] = np.random.uniform(0, 1, param_permutations)
            try:
                refit_fxn_test_output = self.wip_refit(DUMMY_CV_RESULTS)
            except:
                _exc(f"refit callable excepted during function validation. reason={sys.exc_info()[1]}")
            del DUMMY_CV_RESULTS
            _msg = lambda output: (f"if a callable is passed to refit, it must yield or return an integer, and it must be within range "
                f"of cv_results_ rows. the failure has occurred on a randomly filled copy of cv_results that allows testing of the refit "
                f"function before running the entire grid search. the failure may not necessarily occur when cv_results is filled with "
                f"real results.  refit function output = {output}, cv_results rows = {param_permutations}")
            if not int(refit_fxn_test_output) == refit_fxn_test_output:
                _exc(_msg(refit_fxn_test_output))
            if refit_fxn_test_output > param_permutations:
                _exc(_msg(refit_fxn_test_output))
            del refit_fxn_test_output, _msg


        if self._dask_estimator and self.cache_cv:
            CACHE_CV = list(
                _get_kfold(X, self.n_splits_, self.iid, self.verbose, _y=y))
        else:
            CACHE_CV = None


        for trial_idx, _grid in enumerate(self.cv_results_['params']):

            # THRESHOLDS MUST BE IN wip_param_grid(s). REMOVE IT AND SET IT ASIDE, BECAUSE IT CANT BE PASSED TO estimator
            # ONLY DO THIS FIRST TIME ACCESSING A NEW wip_param_grid

            if self.verbose >= 3:
                print(f'\nparam grid {trial_idx+1} of {param_permutations}: {_grid}')

            pgk = PARAM_GRID_KEY[trial_idx]
            if trial_idx==0 or (pgk != PARAM_GRID_KEY[trial_idx-1]):
                THRESHOLDS = THRESHOLD_DICT[trial_idx]
            del pgk

            self._estimator.set_params(**_grid)

            FITTED_ESTIMATOR_COEFS = np.empty(self.n_splits_, dtype=object)  # KEEP THE COEFS IN CASE THEY ARE NEEDED FOR TRAIN SCORING
            TEST_FOLD_FIT_TIME_VECTOR = np.empty(self.n_splits_, dtype=np.float64)
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX = np.ma.empty((self.n_splits_, len(THRESHOLDS), len(self.scorer_)), dtype=np.float64)
            TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX = np.ma.empty((self.n_splits_, len(THRESHOLDS), len(self.scorer_)), dtype=np.float64)
            TRAIN_FOLD_x_SCORER__SCORE_MATRIX = np.ma.empty((self.n_splits_, len(self.scorer_)))

            assert FITTED_ESTIMATOR_COEFS.shape == (self.n_splits_,), \
                "FITTED_ESTIMATOR_COEFS is misshapen"
            assert TEST_FOLD_FIT_TIME_VECTOR.shape == (self.n_splits_,), \
                "TEST_FOLD_FIT_TIME_VECTOR is misshapen"
            assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX.shape == \
                   (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
                "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX is misshapen"
            assert TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX.shape == \
                   (self.n_splits_, len(THRESHOLDS), len(self.scorer_)), \
                "TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX is misshapen"
            assert TRAIN_FOLD_x_SCORER__SCORE_MATRIX.shape == \
                   (self.n_splits_, len(self.scorer_)), \
                "TRAIN_FOLD_x_SCORER__SCORE_MATRIX is misshapen"

            ###################################################################
            # CORE GRID SEARCH ################################################

            # FIT ALL FOLDS ###################################################
            ARGS_FOR_PARALLEL_FIT = (self._estimator, _grid, self.error_score)

            FIT_OUTPUT = list()

            for f_idx, (train_idxs, test_idxs) in enumerate(
                    _get_kfold(X, self.n_splits_, self.iid, self.verbose, _y=y)):
                X_train, y_train = _fold_splitter(train_idxs, test_idxs, X, y)[0::2]
                FIT_OUTPUT.append(
                    _parallelized_fit(f_idx, X_train, y_train, train_idxs, test_idxs, *ARGS_FOR_PARALLEL_FIT, **fit_params))


            del ARGS_FOR_PARALLEL_FIT
            # END FIT ALL FOLDS ###############################################

            # COMPILE INFO FROM FIT ###########################################

            ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS = np.ma.empty((self.n_splits_, 2, X.shape[0]//self.n_splits_ + 1), dtype=np.float64)
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

                    continue   # GO TO NEXT FOLD

                pp_time = time.perf_counter()
                _predict_proba = _estimator_.predict_proba(X_test)[:, -1].ravel()
                try:
                    _predict_proba = _predict_proba.compute(scheduler=self.scheduler)
                except:
                    pass

                if self.verbose >= 5:
                    print(f'fold {f_idx+1} tests predict_proba time = {time.perf_counter() - pp_time: ,.3g} s')
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
                if self.verbose >= 5: print(f'Start scoring with different thresholds and scorers')

                joblib_kwargs = {
                    'backend': 'dask' if self._dask_estimator else 'multiprocessing',
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

                if self.verbose >= 5: print(f'End scoring with different thresholds and scorers')
                test_predict_and_score_tf = time.perf_counter()

                for thresh_idx, RESULT_SET in enumerate(JOBLIB_TEST_SCORE_OUTPUT):
                    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[f_idx, thresh_idx, :] = RESULT_SET[0]
                    TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, thresh_idx, :] = RESULT_SET[1]

                del JOBLIB_TEST_SCORE_OUTPUT


                if self.verbose >= 5:
                    FOLD_TIMES = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX[f_idx, :, :]
                    print(f'fold {f_idx+1} total tests score time = {FOLD_TIMES.sum(): ,.3g} s')
                    print(f'fold {f_idx+1} avg tests score time = {FOLD_TIMES.mean(): ,.3g} s')
                    del FOLD_TIMES
                    print(f'fold {f_idx+1} total tests predict & score wall time = {test_predict_and_score_tf - test_predict_and_score_t0: ,.3g} s')

            del ARRAY_OF_PREDICT_PROBAS_AND_Y_TESTS, LEN_Y_TEST, FIT_EXCEPTED, y_test, _predict_proba
            del test_predict_and_score_t0, test_predict_and_score_tf
            # END GET SCORE FOR ALL THRESHOLDS ################################

            # END CORE GRID SEARCH ############################################
            ###################################################################

            # NEED TO GET BEST THRESHOLDS BEFORE IDENTIFYING BEST SCORES ######

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
            TEST_FOLD_x_SCORER__SCORE_MATRIX = np.ma.empty((self.n_splits_, len(self.scorer_)))

            for scorer_idx, threshold_idx in zip(range(len(self.scorer_)), TEST_BEST_THRESHOLD_IDXS_BY_SCORER):
                TEST_FOLD_x_SCORER__SCORE_MATRIX[:, scorer_idx] = TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX[:, threshold_idx, scorer_idx]

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
                    DUM_DATA = X.loc[:2, :]  # PASS WITH COLUMN NAMES IF WAS A DF TO GET feature_names_in_
                except:
                    DUM_DATA = X[:2, :]
                self._estimator.fit(DUM_DATA, [0, 1])
                del DUM_DATA

                for f_idx, (train_idxs, test_idxs) in enumerate(CACHE_CV or _get_kfold(X, self.n_splits_, self.iid, self.verbose, _y=y, **kfold_params)):
                    # IF FITTED_ESTIMATOR_COEFS[f_idx] IS None, IT IS BECAUSE fit() EXCEPTED ON THAT FOLD.
                    # PUT IN error_score IF error_score!=np.nan, elif is np.nan ALSO MASK THIS FOLD IN
                    # TRAIN_FOLD_x_SCORER__SCORE_MATRIX, AND SKIP TO NEXT FOLD
                    if FITTED_ESTIMATOR_COEFS[f_idx] is None:
                        TRAIN_FOLD_x_SCORER__SCORE_MATRIX[f_idx, :] = self.error_score
                        if self.error_score is np.nan:
                            TRAIN_FOLD_x_SCORER__SCORE_MATRIX[f_idx, :] = np.ma.masked

                        continue

                    X_train, y_train = _fold_splitter(train_idxs, test_idxs, X, y)[0::2]

                    assert X_train.shape[0] == y_train.shape[0], "X_train.shape[0] != y_train.shape[0]"

                    # PUT THE RETAINED COEFS FROM THE fit() EARLIER BACK INTO estimator
                    self._estimator.coef_ = FITTED_ESTIMATOR_COEFS[f_idx]
                    _predict_proba = self._estimator.predict_proba(X_train)[:, -1]
                    try:
                        _predict_proba = _predict_proba.compute(scheduler=self.scheduler)
                    except:
                        pass

                    assert len(_predict_proba) == X_train.shape[0], \
                        "len(_predict_proba) != X_train.shape[0]"

                    train_predict_and_score_t0 = time.perf_counter()
                    train_pred_time, train_score_time = 0, 0
                    for s_idx, scorer_key in enumerate(self.scorer_):
                        train_pred_t0 = time.perf_counter()
                        y_train_pred = (_predict_proba >= THRESHOLDS[TEST_BEST_THRESHOLD_IDXS_BY_SCORER[s_idx]])
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
                        print(f'fold {f_idx+1} total train predict time = {train_pred_time: ,.3g} s')
                        print(f'fold {f_idx+1} total train score time = {train_score_time: ,.3g} s')
                        print(f'fold {f_idx+1} avg train score time = {train_score_time/len(self.scorer_): ,.3g} s')
                        print(f'fold {f_idx+1} total train predict and score time = {time.perf_counter() - train_predict_and_score_t0: ,.3g} s')

                try: del X_train, y_train, _predict_proba, y_train_pred, __, train_predict_and_score_t0, train_pred_time, \
                        train_score_time, train_pred_t0, train_pred_time, train_score_t0, train_score_time
                except: pass

            del FITTED_ESTIMATOR_COEFS, TEST_BEST_THRESHOLD_IDXS_BY_SCORER
            # END SCORE TRAIN FOR THE BEST THRESHOLDS #########################

            if self.verbose >= 5:
                print(f'Start filling cv_results_'); cv_t0 = time.perf_counter()

            # UPDATE cv_results_ WITH RESULTS #################################
            self.cv_results_ = _cv_results_score_updater(
                TEST_FOLD_x_SCORER__SCORE_MATRIX,
                'test',
                trial_idx,
                self.scorer_,
                self.n_splits_,
                self.cv_results_
            )

            if self.return_train_score:
                self.cv_results_ = _cv_results_score_updater(
                    TRAIN_FOLD_x_SCORER__SCORE_MATRIX,
                    'train',
                    trial_idx,
                    self.scorer_,
                    self.n_splits_,
                    self.cv_results_
                )
            # END UPDATE cv_results_ WITH RESULTS #############################

            del TEST_FOLD_x_SCORER__SCORE_MATRIX, TRAIN_FOLD_x_SCORER__SCORE_MATRIX

            # UPDATE cv_results_ WITH TIMES ############################
            for cv_results_column_name in ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']:
                if cv_results_column_name not in self.cv_results_:
                    _exc(f"appending time results to a column in cv_results_ that doesnt exist but should ({cv_results_column_name})")

            self.cv_results_['mean_fit_time'][trial_idx] = np.mean(TEST_FOLD_FIT_TIME_VECTOR)
            self.cv_results_['std_fit_time'][trial_idx] = np.std(TEST_FOLD_FIT_TIME_VECTOR)

            self.cv_results_['mean_score_time'][trial_idx] = np.mean(TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
            self.cv_results_['std_score_time'][trial_idx] = np.std(TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX)
            # END UPDATE cv_results_ WITH TIMES ########################

            if self.verbose >= 5:
                print(f'End filling cv_results_ = {time.perf_counter() - cv_t0: ,.3g} s')

            del TEST_FOLD_FIT_TIME_VECTOR, TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_TIME_MATRIX

        del THRESHOLD_DICT, CACHE_CV

        # FINISH RANK COLUMNS HERE #####################################
        # ONLY DO TEST COLUMNS, DONT DO TRAIN RANK
        for scorer_suffix in self.scorer_:

            if f'rank_test_{scorer_suffix}' not in self.cv_results_:
                _exc(f"appending tests scores to a column in cv_results_ that doesnt exist but should (rank_test_{scorer_suffix})")

            _ = self.cv_results_[f'mean_test_{scorer_suffix}']
            self.cv_results_[f'rank_test_{scorer_suffix}'] = len(_) - ss.rankdata(_, method='max') + 1
            del _
        # END FINISH RANK COLUMNS HERE #################################

        """
        refit:
            if False:
            self.best_index_: Indeterminate, not explicitly stated. Most likely is available.
            self.best_estimator_: estimator --- Not available if refit=False.
            self.best_score_: is available as a float
            self.best_params_: For multi-metric evaluation, not present if refit is not specified.
            self.n_features_in_: Not available.

            if not False:

                expose refit_time_ attr --- Seconds used for refitting the best model on the whole dataset.



                the refitted estimator is made available at the best_estimator_ attribute and
                                permits using predict directly on this GridSearchCV instance.

                one metric:
                    True --- maximum score is used to choose the best estimator
                             best_index_ is available
                             best_estimator is available, set according to the returned best_index_
                             best_score_ is available as a float
                             best_params_ is available, set according to the returned best_index_
                             self.n_features_in_: sklearn only. Only available when refit = True (single metric)

                multi metric:
                    str --- maximum score for that metric is used to choose the best estimator
                            best_index_ is available
                            best_estimator is available, set according to the returned best_index_
                            best_score_ is available as a float!  THE DASK DOCS ARE INCORRECT!  NO DICTIONARY EVER!
                            best_params_ is available, set according to the returned best_index_

                both:
                    a function which returns the selected best_index_ when passed cv_results_
                        ---best_index is available
                        ---best_estimator_ is available, set according to the returned best_index_
                        ---best_score_ attribute will not be available.
                        ---best_params_ is available, set according to the returned best_index_
        """

        """
        self.best_index_: int or dict of ints
        The index of the cv_results_ array which corresponds to the best candidate parameter setting.
        This locates the dict in search.cv_results_['params'][search.best_index_] holding the parameter settings
        for the best model, i.e., the one that gives the highest mean score (search.best_score_) on the holdout data.
        When using multiple metrics, best_index_ will be a dictionary where the keys are the names of the scorers,
        and the values are the index with the best mean score for that scorer, as described above.
        """

        """
        self.best_estimator_: estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss
        if specified) on the left out data. Not available if refit=False.
        """

        """
        self.best_score_: float
            THE DASK DOCS ARE INCORRECT!  NO DICTIONARY EVER!
            Mean tests score of best_estimator on the hold out data.
        """

        """
        self.best_params_: dict
            The dict at search.cv_results_['params'][search.best_index_] that holds the parameter settings that yields
            the best model (i.e, gives the highest mean score -- search.best_score_ -- on the hold out data.)
            For multi-metric evaluation, this is present only if refit is specified.
        """

        """
        self.n_features_in_:
            sklearn only
            Number of features seen during fit. Only available when refit = True.
        """

        """
        self.refit_time_
            Seconds used for refitting the best model on the whole dataset.
            This is present only if refit is not False.
        """

        refit_is_str = isinstance(self.wip_refit, str)
        refit_is_false = self.wip_refit == False
        refit_is_callable = callable(self.wip_refit)

        if refit_is_callable:
            refit_fxn_output = self.wip_refit(deepcopy(self.cv_results_))
            _msg = (f"if a callable is passed to refit, it must yield or return an integer, and it must be within range "
                    f"of cv_results_ rows.")
            if not int(refit_fxn_output) == refit_fxn_output: _exc(_msg)
            if refit_fxn_output > param_permutations:
                _exc(_msg)
            self.best_index_ = refit_fxn_output
            del refit_fxn_output
            self.best_params_ = self.cv_results_['params'][self.best_index_]
            if len(self.scorer_) == 1:
                self.best_threshold_ = self.cv_results_['best_threshold'][self.best_index_]
                self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]
            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass
        elif refit_is_false:
            if len(self.scorer_) == 1:
                self.best_index_ = np.arange(param_permutations)[self.cv_results_['rank_test_score'] == 1][0]
                self.best_params_ = self.cv_results_['params'][self.best_index_]
                self.best_threshold_ = self.cv_results_['best_threshold'][self.best_index_]
                self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]
            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass
        elif refit_is_str:
            # DOESNT MATTER WHAT len(self.scorer_) IS
            self.best_index_ = np.arange(param_permutations)[self.cv_results_[f'rank_test_{self.wip_refit}'] == 1][0]
            self.best_params_ = self.cv_results_['params'][self.best_index_]
            threshold_column = f'best_threshold' if len(self.scorer_)==1 else f'best_threshold_{self.wip_refit}'
            self.best_threshold_ = self.cv_results_[threshold_column][self.best_index_]
            del threshold_column
            self.best_score_ = self.cv_results_[f'mean_test_{self.wip_refit}'][self.best_index_]

        del refit_is_str, refit_is_callable, refit_is_false

        if self.wip_refit:
            if self.verbose >= 3: print(f'\nStarting refit...')
            self.best_estimator_ = self._estimator
            self.best_estimator_.set_params(**self.best_params_)
            t0 = time.perf_counter()

            self.best_estimator_.fit(X, y, **refit_params)
            self.refit_time_ = time.perf_counter() - t0
            del t0
            if self.verbose >= 3: print(f'Finished refit. time = {self.refit_time_}')

            # feature_names_in_: ndarray of shape (n_features_in_,)
            # Names of features seen during fit. Only defined if best_estimator_ is defined
            # and if best_estimator_ exposes feature_names_in_ when fit.
            try: self.feature_names_in_ = self.best_estimator_.feature_names_in_
            except:
                try: self.feature_names_in_ = self._feature_names_in
                except: pass

            # PIZZA CHANGED THIS 24_02_29_12_22_00
            # return self.best_estimator_

        elif self.wip_refit is False:
            # PIZZA CHANGED THIS 24_02_29_12_22_00
            # return self._estimator
            pass

        del _exc

        return self


    def get_params(self, deep:bool=True):

        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Return
        ------
        -
            paramsdict: Parameter names mapped to their values.

        Rules of get_params, for sklearn/dask GridSearchCV, pipelines, and single estimators:
        get_params() always returns a dictionary
        --- single dask/sklearn estimator, deep=True/False is irrelevant
            shallow :return: all the est's args/kwargs
            deep :return: all the est's args/kwargs
        --- sklearn/dask pipeline
            shallow :return: 3 params for the pipeline (steps, memory, verbose)
            deep :return: 3 params for the pipeline + each n object(s) in steps + get_params() for each n object(s) with '{step_name}__' prefix
        --- sklearn/dask GridSearchCV with single dask/sklearn estimator
            shallow :return: 10 sklearn (11 dask) GridSearchCV args/kwargs (which includes estimator)
            deep :return: 10 (11) GSCV args/kwargs (which includes estimator) + get_params() for estimator with 'estimator__' prefix
        --- sklearn/dask GridSearchCV with pipeline
            shallow :return: 10 sklearn (11 dask) GridSearchCV args/kwargs (which includes pipeline)
            deep :return: 10 (11) GSCV args/kwargs (which includes pipeline) + get_params(deep=True) for pipeline (see above) with 'estimator__' prefix
        """

        # sklearn / dask -- this is always available, before & after fit

        paramsdict = {}

        self._is_dask_estimator()   # set self._dask_estimator

        paramsdict['estimator'] = self._estimator
        if self._dask_estimator: paramsdict['cache_cv'] = self.cache_cv
        paramsdict['cv'] = self.cv
        paramsdict['error_score'] = self.error_score
        if self._dask_estimator: paramsdict['iid'] = self.iid
        paramsdict['n_jobs'] = self.n_jobs
        paramsdict['param_grid'] = self.param_grid
        if not self._dask_estimator: paramsdict['pre_dispatch'] = self.pre_dispatch
        paramsdict['refit'] = self.refit
        paramsdict['return_train_score'] = self.return_train_score
        if self._dask_estimator: paramsdict['scheduler'] = self.scheduler
        paramsdict['scoring'] = self.scoring
        # paramsdict['thresholds'] = self.thresholds
        if not self._dask_estimator: paramsdict['verbose'] = self.verbose

        # THIS IS CORRECT FOR BOTH SIMPLE ESTIMATOR OR PIPELINE
        if deep:
            paramsdict = paramsdict | {f'estimator__{k}': v for k, v in self._estimator.get_params(deep=True).items()}

        # ALPHABETIZE paramsdict
        paramsdict = {k: paramsdict.pop(k) for k in sorted(paramsdict)}

        return paramsdict


    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        (can set params of the gridsearch instance and/or the wrapped
        estimator, verified 24_02_16_09_31_00)
        The method works on simple estimators as well as on nested
        objects (such as Pipeline). The latter have parameters of the
        form <component>__<parameter> so that it’s possible to update
        each component of a nested object.

        Parameters
        ----------
        **params: dict[str: any] - Estimator parameters.

        Return
        ------
        -
            GridSearchThresholdCV instance.

        """

        # estimators, pipelines, and gscv all raise exception for invalid
        # keys (parameters) passed

        self._is_dask_estimator()

        est_params = {k.replace('estimator__', ''): v for k,v in params.items() if 'estimator__' in k}
        gstcv_params = {k.lower(): v for k,v in params.items() if 'estimator__' not in k.lower()}

        if 'pipe' in str(type(self._estimator)).lower():
            for _name, _est in self._estimator.steps:
                if f'{_name}' in est_params:
                    self.set_estimator_error()

        if 'estimator' in gstcv_params:
            self.set_estimator_error()

        # IF self._estimator is dask/sklearn est/pipe, THIS SHOULD HANDLE
        # EXCEPTIONS FOR INVALID PASSED PARAMS
        self._estimator.set_params(**est_params)

        for _param in gstcv_params:
            if not hasattr(self, _param):
                __ = 'GridSearchThresholdCV'  # type(self).__name__
                raise Exception(f"Invalid parameter '{_param}' for {__} when in {'dask' if self._dask_estimator else 'sklearn'} mode")

        # THESE WILL BE VALIDATED NEXT TIME validate_and_reset() IS CALLED, WHICH ONLY HAPPENS NEAR THE TOP OF fit()
        if 'param_grid' in gstcv_params: self.param_grid = gstcv_params['param_grid']
        if 'thresholds' in gstcv_params: self.thresholds = gstcv_params['thresholds']
        if 'scoring' in gstcv_params: self.scoring = gstcv_params['scoring']
        if 'n_jobs' in gstcv_params: self.n_jobs = gstcv_params['n_jobs']
        if 'pre_dispatch' in gstcv_params: self.pre_dispatch = gstcv_params['pre_dispatch']
        if 'cv' in gstcv_params: self.cv = gstcv_params['cv']
        if 'refit' in gstcv_params: self.refit = gstcv_params['refit']
        if 'verbose' in gstcv_params: self.verbose = gstcv_params['verbose']
        if 'error_score' in gstcv_params: self.error_score = gstcv_params['error_score']
        if 'return_train_score' in gstcv_params: self.return_train_score = gstcv_params['return_train_score']
        if 'iid' in gstcv_params: self.iid = gstcv_params['iid']
        if 'scheduler' in gstcv_params: self.scheduler = gstcv_params['scheduler']
        if 'cache_cv' in gstcv_params: self.cache_cv = gstcv_params['cache_cv']

        del est_params, gstcv_params

        return self



    def visualize(self, filename=None, format=None):
        """
        STRAIGHT FROM DASK SOURCE CODE:
        Render the task graph for this parameter search using ``graphviz``.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional, default = None.
           The name (without an extension) of the file to write to disk.
           If `filename` is None, no file will be written, and we
           communicate with dot using only pipes.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
           Format in which to write output file.  Default is 'png'.
        **kwargs
           Additional keyword arguments to forward to
           ``dask.dot.to_graphviz``.

        Returns
        -------
        result : IPython.diplay.Image, IPython.display.SVG, or None
           See ``dask.dot.dot_graph`` for more information.
        """

        if not self._dask_estimator:
            raise NotImplementedError(f"Cannot visualize a sklearn estimator")

        self.check_is_fitted()
        # PIZZA FIGURE THIS OUT
        return visualize(self._estimator, filename=filename, format=format)


    # END SKLEARN / DASK GridSearchCV Method ###########################
    ####################################################################


    ####################################################################
    # SUPPORT METHODS ##################################################




    def validate_and_reset(self):

        def _exc(reason):
            raise Exception(f"{reason}")

        # VALIDATE estimator ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        if not 'class' in str(type(self._estimator)).lower():
        # if not inspect.isclass(self._estimator): # pizza
            _exc(f"estimator must be a class instance")

        REQUIRED_METHODS = ['fit', 'set_params', 'get_params', 'predict_proba', 'score']
        _HAS_METHODS = []
        for _method in REQUIRED_METHODS:
            _HAS_METHODS.append(callable(getattr(self._estimator, _method, None)))
        if False in _HAS_METHODS:
            _exc(f"estimator must have the following methods: "
                f"{', '.join(REQUIRED_METHODS)}. passed estimator only has the "
                 f"following methods: {', '.join(list(REQUIRED_METHODS[_HAS_METHODS]))}")
        del REQUIRED_METHODS, _HAS_METHODS


        self._is_dask_estimator()   # set self._dask_estimator


        if not self.new_is_classifier(self._estimator):
            _exc(f"estimator must be a classifier to use threshold. use "
                 f"regular sklearn/dask GridSearch CV for a regressor")

        # END VALIDATE estimator ** ** ** ** ** ** ** ** ** ** ** ** **


        # VALIDATE THRESHOLDS & PARAM_GRID ** ** ** ** ** ** ** ** ** **
        def threshold_checker(_thresholds, is_from_kwargs, idx):
            base_msg = (f"must be (1 - a list-type of 1 or more numbers) "
                        f"or (2 - a single number) and 0 <= number(s) <= 1")
            if is_from_kwargs:
                _msg = f"thresholds passed as a kwarg " + base_msg
            else:
                _msg = f"thresholds passed as param to param_grid[{idx}] " + base_msg
            del base_msg

            if _thresholds is None: _thresholds = np.linspace(0,1,21)
            else:
                try:
                    _thresholds = np.array(list(_thresholds), dtype=np.float64)
                except:
                    try:
                        int(_thresholds); _thresholds = np.array([_thresholds], dtype=np.float64)
                    except:
                        _exc(_msg)

            if len(_thresholds)==0:
                _exc(_msg)

            for _thresh in _thresholds:
                try:
                    int(_thresh)
                except:
                    _exc(_msg)

                if not (_thresh >= 0 and _thresh <= 1):
                    _exc(_msg)

            _thresholds.sort()

            del _msg
            return _thresholds

        _msg = (f"param_grid must be a (1 - dictionary) or (2 - a list of "
                f"dictionaries) with strings as keys and lists as values")

        if self.param_grid is None:
            self.wip_param_grid = [{}]
        elif isinstance(self.param_grid, dict):
            self.wip_param_grid = [self.param_grid]
        else:
            try:
                self.wip_param_grid = list(self.param_grid)
            except:
                _exc(_msg)

        # param_grid must be list at this point
        for grid_idx, _grid in enumerate(self.wip_param_grid):
            if not isinstance(_grid, dict):
                _exc(_msg)
            for k, v in _grid.items():
                if not 'str' in str(type(k)).lower():
                    _exc(_msg)
                try:
                    v = compute(v, scheduler=self.scheduler)[0]
                except:
                    pass
                try:
                    v = list(v)
                except:
                    _exc(_msg)
                self.wip_param_grid[grid_idx][k] = np.array(v)

        del _msg

        # at this point param_grid must be a list of dictionaries with
        # str as keys and eager lists as values
        for grid_idx, _grid in enumerate(self.wip_param_grid):

            if np.fromiter(map(lambda x: 'threshold' in x, list(map(str.lower, _grid.keys()))), dtype=bool).sum() > 1:
                _exc(f"there are multiple keys in param_dict[{grid_idx}] indicating threshold")

            new_grid = {}
            for _key, _value in _grid.items():
                if 'threshold' in _key.lower():
                    new_grid['thresholds'] = _value
                else:
                    new_grid[_key] = _value

            _grid = new_grid
            del new_grid

            if 'thresholds' in _grid:
                _grid['thresholds'] = threshold_checker(_grid['thresholds'], False, grid_idx)

            elif 'thresholds' not in _grid:
                _grid['thresholds'] = threshold_checker(self.thresholds, True, None)

            self.wip_param_grid[grid_idx] = _grid

        del threshold_checker, _grid
        # END VALIDATE THRESHOLDS & PARAM_GRID ** ** ** ** ** ** ** ** *

        # VALIDATE scoring / BUILD self.scorer_ ** ** ** ** ** ** ** **
        """
        scoring:
        Strategy to evaluate the performance of the cross-validated model on the tests set.
        If scoring represents a single score, one can use:
            a single string;
            a callable with signature (y_true, y_pred) that returns a single value.
        If scoring represents multiple scores, one can use:
            a list-type of unique strings;
            a list-type of (callable with signature (y_true, y_pred) that returns a single value);
            a dictionary with metric names as keys and callable(y_true, y_pred) as values;
            a callable returning (a dictionary with metric names as keys and callable(y_true, y_pred) as values)
        """

        ALLOWED_SCORING_DICT: ScorerWIPType  = {
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score,
            'average_precision': average_precision_score,
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }


        def string_validation(_string:str):
            _string = _string.lower()
            if _string not in ALLOWED_SCORING_DICT:
                if 'roc_auc' in _string or 'average_precision' in _string:
                    _exc(f"Dont need to use GridSearchThreshold when "
                         f"scoring is roc_auc or average_precision (auc_pr). "
                         f"Use regular dask/sklearn GridSearch and use max(tpr-fpr) "
                         f"to find best threshold for roc_auc, "
                         f"or use max(f1) to find the best threshold for "
                         f"average_precision.")
                else:
                    raise NotImplementedError(
                        f"When specifying scoring by scorer name, must be "
                        f"in {', '.join(list(ALLOWED_SCORING_DICT))} ('{_string}')")

            return _string


        def check_callable_is_valid_metric(fxn_name:str, _callable:callable):
            _truth = np.random.randint(0, 2, (100,))
            _pred = np.random.randint(0, 2, (100,))
            try:
                _value = _callable(_truth, _pred)
            except:
                _exc(f"scoring function '{fxn_name}' excepted during validation")
            if not (_value >= 0 and _value <= 1):
                _exc(f"metric scoring function must have (y_true, y_pred) "
                     f"signature and return a number 0 <= number <= 1")
            del _value


        _msg = (f"scoring must be "
                f"\n1) a single string, or "
                f"\n2) a callable(y_true, y_pred) that returns a single value and 0 <= value <= 1, or "
                f"\n3) a list-type of strings, or "
                f"\n4) a dict of: (metric name: callable(y_true, y_pred), ...)."
                f"\nCannot pass None or bool. Cannot use estimator's default scorer.")

        _scoring = self.scoring

        try:
            _scoring = compute(_scoring, scheduler=self.scheduler)[0]
        except:
            pass

        if isinstance(_scoring, str):
            _scoring = string_validation(_scoring)
            _scoring = {_scoring: ALLOWED_SCORING_DICT[_scoring]}

        elif isinstance(_scoring, (list,tuple,set,np.ndarray)):
            try:
                _scoring = np.array(_scoring)
            except:
                _exc(_msg)
            _scoring = list(_scoring.ravel())
            if len(_scoring)==0:
                _exc(f'scoring is empty --- ' + _msg)
            for idx, string_thing in enumerate(_scoring):
                if not isinstance(string_thing, str):
                    _exc(_msg)
                _scoring[idx] = string_validation(string_thing)

            _scoring = list(set(_scoring))

            _scoring = {k:v for k,v in ALLOWED_SCORING_DICT.items() if k in _scoring}

        elif isinstance(_scoring, dict):
            if len(_scoring)==0:
                _exc(f'scoring is empty --- ' + _msg)
            for key in _scoring:
                # DONT USE string_validation() HERE, USER-DEFINED CALLABLES CAN HAVE USER-DEFINED NAMES
                new_key = key.lower()
                _scoring[new_key] = _scoring.pop(key)
                check_callable_is_valid_metric(new_key, _scoring[new_key])
            del new_key

        elif callable(_scoring):
            check_callable_is_valid_metric(f'score', _scoring)
            _scoring = {f'score': _scoring}

        else:
            _exc(_msg)

        del _msg
        del string_validation, check_callable_is_valid_metric

        """
        dict of functions - Scorer function used on the held out data to 
        choose the best parameters for the model.
        A dictionary of {scorer_name: scorer} when one or multiple metrics 
        are used.
        """

        self.scorer_ = _scoring
        del _scoring

        self.multimetric_ = False if len(self.scorer_) == 1 else True

        del ALLOWED_SCORING_DICT
        # END VALIDATE scoring / BUILD self.scorer_ ** ** ** ** ** ** **

        # VALIDATE n_jobs ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        if not self.n_jobs in ([-1] + list(range(1, 17)) + [None]):
            _exc(f"n_jobs must be an integer or None")
        # END VALIDATE n_jobs ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # VALIDATE pre_dispatch ** ** ** ** ** ** ** ** ** ** ** ** ** *
        if self._dask_estimator:
            try: del self.pre_dispatch
            except: pass
        else:
            # PIZZA FIGURE THIS OUT AND FIX IT
            pass
        # END VALIDATE pre_dispatch ** ** ** ** ** ** ** ** ** ** ** **

        # VALIDATE cv (n_splits_) ** ** ** ** ** ** ** ** ** ** ** ** **
        # UPHOLD THE dask/sklearn PRECEDENT THAT n_splits_ IS NOT AVAILABLE
        # AFTER init(), BUT IS AFTER fit()
        self.n_splits_ = self.cv or 5
        if not self.n_splits_ in range(2, 101): _exc(f"cv must be an integer in range(2,101)")
        # END VALIDATE cv ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        self.wip_refit = self.refit

        try:
            self.wip_refit = compute(self.wip_refit, scheduler=self.scheduler)[0]
        except:
            pass

        _msg = (
            f"refit must be \n1) bool, \n2) None, \n3) a scoring "
            f"metric name, \n4) a callable that returns an integer, or "
            f"\n5) a list-type of length 1 that contains one of those "
            f"four."
        )

        if isinstance(self.wip_refit, dict):
            _exc(_msg)

        def type_getter(refit_arg, _return):
            _is_bool = isinstance(refit_arg, bool)
            _is_str = isinstance(refit_arg, str)
            _is_none = refit_arg is None
            _is_callable = callable(refit_arg)
            if not _return:
                return (_is_bool or _is_str or _is_none or _is_callable)
            return _is_bool, _is_str, _is_none, _is_callable

        if type_getter(self.wip_refit, False):
            _is_bool, _is_str, _is_none, _is_callable = type_getter(self.wip_refit, True)
        else:
            try:
                self.wip_refit = list(self.wip_refit)
                if len(self.wip_refit) > 1:
                    _exc(_msg)
                else:
                    self.wip_refit = self.wip_refit[0]
                if type_getter(self.wip_refit, False):
                    _is_bool, _is_str, _is_none, _is_callable = type_getter(self.wip_refit, True)
                else:
                    _exc(_msg)
            except:
                _exc(_msg)

        del type_getter

        if self.wip_refit is None:
            self.wip_refit = False
            _is_bool = True
            del _is_none

        if self.wip_refit is False or _is_callable:
            # CANT VALIDATE CALLABLE OUTPUT HERE, cv_results_ IS NOT AVAILABLE

            # AS OF 24_02_25_12_22_00 THERE ISNT A WAY TO RETURN A best_threshold_ WHEN MULTIPLE SCORERS ARE PASSED
            # TO scoring AND refit IS False OR A CALLABLE (OK IF REFIT IS A STRING).  IF THIS EVER CHANGES, THIS
            # WARNING CAN COME OUT.
            if len(self.scorer_) > 1:
                warnings.warn(
                    f"WHEN MULTIPLE SCORERS ARE USED:\n"
                    f"Cannot return a threshold if refit is False or callable.\n"
                    f"If refit is False: best_index_, best_estimator_, best_score_, and best_threshold_ are not available.\n"
                    f"if refit is callable: best_score_ and best_threshold_ are not available.\n"
                    f"In either case, access score and threshold info via the cv_results_ attribute."
                )

            del _is_bool, _is_str, _is_callable
            pass
        else:  # refit CAN BE True OR (MATCHING A STRING IN scoring) ONLY
            refit_is_true = self.wip_refit==True
            refit_is_str = _is_str
            del _is_bool, _is_str, _is_callable

            _msg = lambda x: f"egregious coding failure - refit_is_str and refit_is_bool are both {x}"
            if refit_is_str and refit_is_true:
                _exc(_msg(True))
            elif not refit_is_str and not refit_is_true:
                _exc(_msg(False))
            del _msg

            if refit_is_str:
                self.wip_refit = self.wip_refit.lower()

            # self.scorer_ KEYS CAN ONLY BE SINGLE STRINGS: user-defined via dict, 'score', or actual score method name
            if refit_is_true:
                if len(self.scorer_) == 1:
                    self.wip_refit = 'score'
                elif len(self.scorer_) > 1:
                    _exc(f"when scoring is multiple metrics, refit must be: "
                         f"\n1) a single string exactly matching a scoring method in scoring, "
                         f"\n2) a callable that returns an index in cv_results_ to use as best_index_, "
                         f"\n3) False")
            elif refit_is_str:
                if self.wip_refit not in self.scorer_:
                    if len(self.scorer_) == 1:
                        _exc(f"if using a single scoring metric, the allowed entries for refit are True, False, a callable "
                            f"that returns a best_index_, or a string that exactly matches the string passed to scoring")
                    elif len(self.scorer_) > 1:
                        _exc(f"if refit is a string, refit must exactly match one of the scoring methods in scoring")
                elif len(self.scorer_) == 1:
                    self.wip_refit = 'score'

            del refit_is_true, refit_is_str

        # IF AN INSTANCE HAS ALREADY BEEN fit() WITH refit != False,
        # POST-REFIT ATTRS WILL BE AVAILABLE. BUT IF SUBSEQUENTLY refit
        # IS SET TO FALSE VIA set_params, THE POST-REFIT ATTRS NEED TO BE
        # REMOVED. IN THIS CONFIGURATION (i) THE REMOVE WILL HAPPEN AFTER
        # fit() IS CALLED AGAIN WHERE THIS METHOD (val&reset) IS CALLED
        # NEAR THE TOP OF fit() (ii) SETTING NEW PARAMS VIA set_params()
        # WILL LEAVE POST-REFIT ATTRS AVAILABLE ON AN INSTANCE THAT SHOWS
        # CHANGED PARAM SETTINGS (AS VIEWED VIA get_params()) UNTIL fit()
        # IS RUN.
        if self.wip_refit is False:
            try:
                del self.best_estimator_
            except:
                pass
            try:
                del  self.refit_time_
            except:
                pass
            try:
                del self.feature_names_in_
            except:
                pass

        # END VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # NOW THAT refit IS VALIDATED, IF ONE THING IN SCORING, CHANGE THE KEY TO 'score'
        if len(self.scorer_)==1:
            self.scorer_ = {'score':v for k,v in self.scorer_.items()}

        # VALIDATE verbose ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if not self._dask_estimator:
            _msg = f"verbose must be a bool or a numeric > 0"
            if not isinstance(self.verbose, bool) and not True in [x in str(type(self.verbose)).lower() for x in ['int', 'float']]:
                _exc(_msg)
            elif self.verbose < 0:
                _exc(_msg)
            del _msg
            if self.verbose is True:
                self.verbose = 10
        # END VALIDATE verbose ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # VALIDATE error_score ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if isinstance(self.error_score, str):
            self.error_score = self.error_score.lower()
            if not self.error_score=='raise':
                _exc(f"the only string that can be passed to kwarg error_score is 'raise'")
        else:
            _msg = f"kwarg error_score must be 1) 'raise', 2) a number 0 <= number <= 1, 3) np.nan"
            if self.error_score is np.nan:
                pass
            else:
                try:
                    np.float64(self.error_score)
                except:
                    _exc(_msg)
                if not (self.error_score>=0 and self.error_score<=1):
                    _exc(_msg)
            del _msg
        # END VALIDATE error_score ** ** ** ** ** ** ** ** ** ** ** ** *

        # VALIDATE return_train_score ** ** ** ** ** ** ** ** ** ** ** *
        if self.return_train_score is None:
            self.return_train_score=False
        if not isinstance(self.return_train_score, bool):
            _exc(f"return_train_score must be True, False, or None")
        # END VALIDATE return_train_score ** ** ** ** ** ** ** ** ** **

        # OTHER POSSIBLE KWARGS FOR DASK SUPPORT
        # VALIDATE iid ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        if self._dask_estimator:
            if not isinstance(self.iid, bool):
                _exc(f'kwarg iid must be a bool')
        else:
            pass
        # END VALIDATE iid ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # VALIDATE scheduler ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        if self._dask_estimator:
            if self.scheduler is None:
                # If no special scheduler is passed, use a n_jobs local cluster
                self.scheduler = Client(n_workers=self.n_jobs, threads_per_worker=1, set_as_default=True)
            else:
                # self.scheduler ONLY FLOWS THRU TO compute(), SO LET compute() HANDLE VALIDATION
                pass
        else:
            try:
                del self.scheduler
            except:
                pass
        # END VALIDATE scheduler ** ** ** ** ** ** ** ** ** ** ** ** **

        # VALIDATE cache_cv ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if self._dask_estimator:
            if not isinstance(self.cache_cv, bool):
                _exc(f'kwarg cache_cv must be a bool')
        else:
            try:
                del self.cache_cv
            except:
                pass
        # END VALIDATE cache_cv ** ** ** ** ** ** ** ** ** ** ** ** ** *

        self.fit_excepted = False

        del _exc

    # END validate_and_reset ###########################################


    def _is_dask_estimator(self):
        self._dask_estimator = _is_dask_estimator(self._estimator)


    @staticmethod
    def new_is_classifier(estimator_):
        # pizza, new_is_classifier can probably come out in the future
        # and just directly drop in pybear_is_classifier
        return pybear_is_classifier(estimator_)


    # END SUPPORT METHODS ##############################################
    ####################################################################





















