# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import warnings
from typing import Union

import numpy as np
import dask.array as da
import distributed

from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             average_precision_score,
                             f1_score, precision_score,
                             recall_score
                             )

from model_selection.GSTCV._type_aliases import (
    ScorerInputType, ScorerWIPType,
    ClassifierProtocol, ParamGridType
)

from pybear.base import is_classifier as pybear_is_classifier


class _GSTCVMixin:

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
        thresholds: Union[np.ndarray[int,float],list[int, float],int,float]=None,  # pizza
        scoring: ScorerInputType=None,
        n_jobs: Union[int,None]=1,
        pre_dispatch: Union[str,None]='2*n_jobs',
        cv: Union[int,None]=None,
        refit: Union[callable, bool, str, list[str], None]=True,
        verbose: Union[int, bool]=0,
        error_score: Union[str,int,float]=np.nan,  # pizza
        return_train_score: bool=False,
        # OTHER POSSIBLE KWARGS FOR DASK SUPPORT
        iid: bool=True,
        scheduler: distributed.scheduler.Scheduler=None,
        cache_cv:bool=True
        ):


        self._estimator = estimator
        self.param_grid = param_grid
        self.thresholds = thresholds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.iid = iid
        self.scheduler = scheduler
        self.cache_cv = cache_cv

    # END init ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self.set_estimator_error()


    @property
    def classes_(self):
        """
        classes_: ndarray of shape (n_classes,) - Class labels. Only
            available when refit is not False and the estimator is a
            classifier (must be classifier --- validated in
            validate_and_reset())


        """
        self.check_refit_is_false_no_attr_no_method('classes_')

        if not hasattr(self, '_classes'):   # MEANS fit() HASNT BEEN RUN
            self.estimator_hasattr('classes_')   # GET THE EXCEPTION

        if self._classes is None:
            # self._classes IS A HOLDER THAT IS FILLED ONE TIME WHEN
            # THIS METHOD IS CALLED
            # self._classes_ IS y AND CAN ONLY BE np OR da ARRAY
            if hasattr(self._estimator, 'classes_'):
                self._classes = self.best_estimator_.classes_
            else:
                # 24_02_25_16_13_00 --- GIVES USER ACCESS TO THIS ATTR FOR DASK
                # da.unique WORKS ON np AND dask arrays
                self._classes = da.unique(self._classes_).compute()

            del self._classes_

        return self._classes


    @property
    def n_features_in_(self):
        """
            property n_features_in_: Number of features seen during fit.
            Only available when refit is not False.

        """

        try: self.check_is_fitted()
        except:
            __ = 'GridSearchThresholdCV'  # type(self).__name__
            raise Exception(f"{__} object has no n_features_in_ attribute.")

        if self._n_features_in is None:
            # self._n_features_in IS A HOLDER THAT IS FILLED ONE TIME
            # WHEN THIS METHOD IS CALLED
            # self._n_features_in_ IS X.shape[0] AND CAN BE int OR delayed
            if hasattr(self._estimator, 'n_features_in_'):
                self._n_features_in = self.best_estimator_.n_features_in_
            elif self.wip_refit is not False:
                # 24_02_25_17_07_00 --- GIVES USER ACCESS TO THIS ATTR FOR DASK
                try: self._n_features_in = self._n_features_in_.compute()
                except: self._n_features_in = self._n_features_in_

            del self._n_features_in_

        return self._n_features_in


    ####################################################################
    # SKLEARN / DASK GridSearchCV Methods ##############################

    def decision_function(self, X):
        """
        Call decision_function on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports decision_function.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_score: ndarray of shape (n_samples,) or (n_samples, n_classes) or
            (n_samples, n_classes * (n_classes-1) / 2) - Result of the
            decision function for X based on the estimator with the best
            found parameters.


        """

        self.estimator_hasattr('predict_proba')

        self.check_refit_is_false_no_attr_no_method('predict_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.decision_function(X)


    def get_metadata_routing(self):
        """
        Get metadata routing of this object.
        Please check User Guide on how the routing mechanism works.

        Return
        ------
        -
            routingMetadataRouter: A MetadataRouter encapsulating routing
                information.
        """

        # sklearn only --- always available, before and after fit()

        __ = 'GridSearchThresholdCV'  # type(self).__name__
        raise NotImplementedError(f"get_metadata_routing is not implemented in {__}.")


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


    def inverse_transform(self, Xt):
        """
        Call inverse_transform on the estimator with the best found params.
        Only available if the underlying estimator implements
        inverse_transform and refit is not False.

        Parameters
        ----------
        Xt: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        -
            X Union[ndarray, sparse matrix, pizza] of shape (n_samples,
                n_features) - Result of the inverse_transform function
                for Xt based on the estimator with the best found parameters.


        """

        # PIZZA 24_07_01_08_44_00 FIGURE OUT IF THIS SHOULD STAY OR GO

        self.estimator_hasattr('inverse_transform')

        self.check_refit_is_false_no_attr_no_method('inverse_transform')

        return self.best_estimator_.inverse_transform(Xt)


    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Parameters
        ----------
        X: pizza

        Return
        ------
        pizza
            The predicted labels or values for X based on the estimator
            with the best found parameters.

        """

        self.estimator_hasattr('predict')

        self.check_refit_is_false_no_attr_no_method('predict')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return (self.best_estimator_.predict_proba(X)[:, -1] > self.best_threshold_).astype(np.uint8)


    def predict_log_proba(self, X):
        """
        Call predict_log_proba on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports predict_log_proba.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_pred: ndarray of shape (n_samples,) or (n_samples, n_classes) -
            Predicted class log-probabilities for X based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute classes_.
        """

        self.estimator_hasattr('predict_log_proba')

        self.check_refit_is_false_no_attr_no_method('predict_log_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.predict_log_proba(X)


    def predict_proba(self, X):
        """
        Call predict_proba on the estimator with the best found parameters.
        Only available if refit is not False and the underlying estimator
        supports predict_proba.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_pred: ndarray of shape (n_samples,) or (n_samples, n_classes) -
            Predicted class probabilities for X based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute classes_.


        """

        self.estimator_hasattr('predict_proba')

        self.check_refit_is_false_no_attr_no_method('predict_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.predict_proba(X)


    def score(self, X, y=None, **params):
        """
        Return the score on the given data, if the estimator has been
        refit. This uses the score defined by scoring where provided, and
        the best_estimator_.score method otherwise.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) - Input data,
            where n_samples is the number of samples and n_features is
            the number of features.
        y: array-like of shape (n_samples, n_output) or (n_samples,),
            default=None - Target relative to X for classification or
            regression; None for unsupervised learning.
        **params: dict of parameters to be passed to the underlying
            scorer(s). Only available if enable_metadata_routing=True.
            See Metadata Routing User Guide for more details.

        Return
        ------
        score: float - The score defined by scoring if provided, and the
            best_estimator_.score method otherwise.
        """

        self.estimator_hasattr('score')

        self.check_refit_is_false_no_attr_no_method('score')

        self.check_is_fitted()

        X, y = self._handle_X_y(X, y=y)[[0,1]]

        return self.scorer_[self.wip_refit](y, (self.predict_proba(X)[:,-1] >= self.best_threshold_), **params)


    def score_samples(self, X):
        """Call score_samples on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports score_samples. New in version 0.24.

        Parameters
        ----------
        X: iterable - Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Return
        ------
        -
            The best_estimator_.score_samples method.

        """

        self.estimator_hasattr('score_samples')

        self.check_refit_is_false_no_attr_no_method('score_samples')

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.score_samples(X)


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


    def transform(self, X):
        """
        Call transform on the estimator with the best found parameters.
        Only available if the underlying estimator supports transform and
        refit is not False.

        Parameters
        ----------
        X: indexable, length n_samples. Must fulfill the input assumptions
            of the underlying estimator.

        Return
        ------
        Xt: {ndarray, sparse matrix} of shape (n_samples, n_features) -
            X transformed in the new space based on the estimator with
            the best found parameters.


        """


        self.estimator_hasattr('transform')

        self.check_refit_is_false_no_attr_no_method('transform')

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.transform(X)


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
            # pizza hashed this 24_06_28_16_59_00 in order to pass
            # self.iid to _get_kfold as an arg
            # try:
            #     del self.iid
            # except:
            #     pass
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


    def set_estimator_error(self):
        __ = 'GridSearchThresholdCV' # type(self).__name__
        raise AttributeError(
            f"Even though sklearn allows it, do not change estimators in "
            f"an instance of {__}. This could only lead to disaster "
            f"because of the sklearn/dask interoperability of {__}. " + \
            f"Create a new instance of {__} with a new estimator."
        )


    def _is_dask_estimator(self):
        self._dask_estimator = _is_dask_estimator(self._estimator)


    @staticmethod
    def new_is_classifier(estimator_):
        # pizza, new_is_classifier can probably come out in the future
        # and just directly drop in pybear_is_classifier
        return pybear_is_classifier(estimator_)


    def estimator_hasattr(self, attr_or_method_name):
        if not hasattr(self._estimator, attr_or_method_name):
            raise AttributeError(f"'{type(self._estimator).__name__}' object has no attribute '{attr_or_method_name}'")
        else:
            return True


    def check_refit_is_false_no_attr_no_method(self, attr_or_method_name):
        if not self.refit:
            raise AttributeError(f"This GridSearchCV instance was initialized "
                f"with `refit=False`. {attr_or_method_name} is available only "
                f"after refitting on the best parameters. You can refit an "
                f"estimator manually using the `best_params_` attribute")
        else:
            return True


    def check_is_fitted(self):

        if not hasattr(self, 'wip_refit'):
            class NotFittedError(Exception):
                pass
            raise NotFittedError(f"This GridSearchCV instance is not "
                f"fitted yet. Call 'fit' with appropriate arguments "
                f"before using this estimator.")
        else:
            return True

    # END SUPPORT METHODS ##############################################
    ####################################################################





















