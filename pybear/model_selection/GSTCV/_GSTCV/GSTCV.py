# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
from typing import Union, Literal, Iterable



from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    ScorerInputType,
    RefitType,
    ClassifierProtocol,
    ParamGridType
)


from model_selection.GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from ._validation._estimator import _validate_estimator
from ._validation._pre_dispatch import _validate_pre_dispatch
from ._validation._verbose import _validate_verbose


from ._handle_X_y._handle_X_y_sklearn import _handle_X_y_sklearn

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder import \
    _cv_results_builder

from model_selection.GSTCV._fit_shared._verify_refit_callable import \
    _verify_refit_callable

from model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit

# pizza! 24_07_14..... prevent [] from getting into _core_fit.... this may
# be done already with a len(param_grid) but make sure.


class GSTCV(_GSTCVMixin):

    """

    24_07_10..... pizza! remember that GSTCV takes metrics, not scorers!
    (where sklearn GSCV needs metrics to be wrapped in make_scorer)

    --- Classifer must have predict_proba method. If does not have predict_proba,
    try to wrap with CalibratedClassifierCV.

    --- 24_07_10 ... reset the estimator to the first-seen params at every transition
        # to a new param grid, and then set the new params as called out
        # in cv_results_. in that way, the user can assume that params
        # not explicitly declared in a param grid are running at their
        # defaults (or whatever values they were hard-coded in when the
        # estimator was instantiated.)


    Notes
    -----
    The parameters selected are those that maximize the score of the left
    out data, unless an explicit score is passed in which case it is
    used instead.  ???




    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn import svm, datasets

    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
    >>> svc = svm.SVC()

    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    # pizza fix
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


    def __init__(
        self,
        estimator: ClassifierProtocol,
        param_grid: ParamGridType,
        *,
        # thresholds can be a single number or list-type passed in
        # param_grid or applied universally via thresholds kwarg
        thresholds: Union[Iterable[Union[int, float]], int, float, None]=None,
        scoring: ScorerInputType='accuracy',
        n_jobs: Union[int,None]=None,
        refit: RefitType=True,
        cv: Union[int,None]=None,
        verbose: Union[int, float, bool]=0,
        pre_dispatch: Union[str, None] = '2*n_jobs',
        error_score: Union[Literal['raise'], int, float] = 'raise',
        return_train_score: bool=False
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

    # END init ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def _handle_X_y(self, X, y=None):

        # PIZZA PIZZA! WHAT ABOUT BLOCKING NON-[0,1] y? THINK ON OVR 🤔

        return _handle_X_y_sklearn(X, y=y)

    ####################################################################
    # SKLEARN GridSearchCV Methods #####################################

    def fit(
            self,
            X: XInputType,
            y: YInputType=None,
            **params
        ):

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


        # DONT unique().compute() HERE, JUST RETAIN THE VECTOR & ONLY DO
        # THE PROCESSING IF classes_ IS CALLED
        self._classes_ = y

        # THIS IS A HOLDER THAT IS FILLED ONE TIME WHEN THE unique().compute()
        # IS DONE ON self._classes_
        self._classes = None


        self.validate_and_reset()

        # BEFORE RUNNING cv_results_builder, THE THRESHOLDS MUST BE
        # REMOVED FROM EACH PARAM GRID IN wip_param_grids
        # BUT THEY NEED TO BE RETAINED FOR CORE GRID SEARCH.
        # pop IS INTENTIONALLY USED HERE TO REMOVE 'thresholds' FROM
        # PARAM GRIDS.
        # 'thresholds' MUST BE REMOVED FROM PARAM GRIDS BEFORE GOING
        # TO _cv_results_builder OR THRESHOLDS WILL BECOME PART OF THE
        # GRID SEARCH, AND ALSO CANT BE PASSED TO estimator.
        THRESHOLD_DICT = {}
        for i in range(len(self.wip_param_grid)):
            THRESHOLD_DICT[i] = self.wip_param_grid[i].pop('thresholds')

        # this could have been at the top of _core_fit but is outside
        # because cv_results is used to validate the refit callable
        # before starting the fit.

        # pizza 24_07_10, for both sk and dask, n_splits_ is only
        # available after fit(). n_splits_ is always returned as a number
        self.cv_results_, PARAM_GRID_KEY = \
            _cv_results_builder(self.wip_param_grid, self.n_splits_,
                                self.scorer_, self.return_train_score
        )

        # USE A DUMMIED-UP cv_results TO TEST IF THE refit CALLABLE RETURNS
        # A GOOD INDEX NUMBER, BEFORE RUNNING THE WHOLE GRIDSEARCH
        if callable(self.wip_refit):
            _verify_refit_callable(self.wip_refit, deepcopy(self.cv_results_))

        self.cv_results_ = _core_fit(
            _X,
            _y,
            self._estimator,
            self.cv_results_,
            self._cv,
            self._error_score,
            self._verbose,
            self.scorer_,
            self._n_jobs,
            self._return_train_score,
            PARAM_GRID_KEY,
            THRESHOLD_DICT,
            **params
        )


        del THRESHOLD_DICT, PARAM_GRID_KEY

        super().fit()

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

        # pizza 24_07_13, what about using set_attr instead?

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


    # END SKLEARN / DASK GridSearchCV Method ###########################
    ####################################################################


    ####################################################################
    # SUPPORT METHODS ##################################################




    def validate_and_reset(self):

        super().validate_and_reset()

        _validate_estimator(self._estimator)

        self._pre_dispatch = _validate_pre_dispatch(self.pre_dispatch)

        self._verbose = _validate_verbose(self.verbose)

    # END SUPPORT METHODS ##############################################
    ####################################################################





















