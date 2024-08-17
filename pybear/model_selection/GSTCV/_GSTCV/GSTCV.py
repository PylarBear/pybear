# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Union, Literal, Iterable, Optional
from contextlib import nullcontext


from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    ScorerInputType,
    RefitType,
    ParamGridType
)


from model_selection.GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from ._validation._estimator import _validate_estimator
from ._validation._pre_dispatch import _validate_pre_dispatch



from ._handle_X_y._handle_X_y_sklearn import _handle_X_y_sklearn


from model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit

# pizza! 24_07_14.... prevent empty param_grid from getting into _core_fit....
# this may be done already with a len(param_grid) but make sure.


class GSTCV(_GSTCVMixin):

    """

    24_07_24..... pizza! remember that GSTCV takes metrics (not scorers!)
    that have signature (y_true, y_pred) and return a single number (where
    sklearn GSCV needs metrics to be wrapped in make_scorer). GSTCV cannot
    directly accept scorer kwargs and pass them to scorers; to pass kwargs
    to your scoring metric, create a wrapper with signature (y_true, y_pred)
    around the metric and hard-code the kwargs into the metric, E.g.:
    def your_metric_wrapper(y_true, y_pred):
        return your_metric(y_true, y_pred, **hard_coded_kwargs)


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
        estimator,
        param_grid: ParamGridType,
        *,
        # thresholds can be a single number or list-type passed in
        # param_grid or applied universally via thresholds kwarg
        thresholds: Optional[Union[Iterable[Union[int, float]], int, float, None]]=None,
        scoring: Optional[ScorerInputType]='accuracy',
        n_jobs: Optional[Union[int,None]]=None,
        refit: Optional[RefitType]=True,
        cv: Optional[Union[int,None]]=None,
        verbose: Optional[Union[int, float, bool]]=0,
        pre_dispatch: Optional[Union[str, None]]='2*n_jobs',
        error_score: Optional[Union[Literal['raise'], int, float]]='raise',
        return_train_score: Optional[bool]=False
    ):



        self.estimator = estimator
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


    ####################################################################
    # SUPPORT METHODS ##################################################

    def _handle_X_y(self, X, y=None):

        # PIZZA PIZZA! WHAT ABOUT BLOCKING NON-[0,1] y? THINK ON OVR 🤔

        return _handle_X_y_sklearn(X, y=y)


    def _core_fit(
            self,
            X: XInputType,
            y: YInputType=None,
            **params
        ):


        self.cv_results_ = _core_fit(
            X,
            y,
            self._estimator,
            self.cv_results_,
            self._cv,
            self._error_score,
            self._verbose,
            self.scorer_,
            self._n_jobs,
            self._return_train_score,
            self._PARAM_GRID_KEY,
            self._THRESHOLD_DICT,
            **params
        )

        return




    def _validate_and_reset(self):

        super()._validate_and_reset()

        _validate_estimator(self.estimator)
        self._estimator = self.estimator

        self._pre_dispatch = _validate_pre_dispatch(self.pre_dispatch)

        self._scheduler = nullcontext()


    # END SUPPORT METHODS ##############################################
    ####################################################################





















