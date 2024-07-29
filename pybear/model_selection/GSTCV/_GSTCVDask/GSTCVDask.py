# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
from typing import Union, Literal, Iterable

import dask, distributed

from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    ScorerInputType,
    RefitType,
    ClassifierProtocol,
    ParamGridType,
    SchedulerType
)

from model_selection.GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from ._validation._cache_cv import _validate_cache_cv
from ._validation._scheduler import _validate_scheduler
from ._validation._iid import _validate_iid
from ._validation._dask_estimator import _validate_dask_estimator

from ._handle_X_y._handle_X_y_dask import _handle_X_y_dask

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder import \
    _cv_results_builder

from model_selection.GSTCV._fit_shared._verify_refit_callable import \
    _verify_refit_callable

from model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit




class GSTCVDask(_GSTCVMixin):

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


    Parameters
    ----------


    cache_cv:
        Union[int, Iterable[tuple[Iterable[int], Iterable[int]]]] -
        *** DIRECTLY FROM DASK DOCS 24_07_18_08_48_00 *** *** *** *** ***
        Whether to extract each train/test subset at most once in each worker
        process, or every time that subset is needed. Caching the splits can
        speedup computation at the cost of increased memory usage per worker
        process.

        If True, worst case memory usage is ``(n_splits + 1) * (X.nbytes +
        y.nbytes)`` per worker. If False, worst case memory usage is
        ``(n_threads_per_worker + 1) * (X.nbytes + y.nbytes)`` per worker.
        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***





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


    def __init__(self,
        estimator,
        param_grid: ParamGridType,
        *,
        # thresholds can be a single number or list-type passed in
        # param_grid or applied universally via thresholds kwarg
        thresholds: Union[Iterable[Union[int, float]], int, float, None]=None,
        scoring: ScorerInputType='accuracy',
        iid: bool = True,
        refit: RefitType = True,
        cv: Union[int, None] = None,
        verbose: Union[int, float, bool] = 0,
        error_score: Union[Literal['raise'], int, float] = 'raise',
        return_train_score: bool = False,
        scheduler: SchedulerType = None,
        n_jobs: Union[int,None]=None,
        cache_cv:bool=True
        ):


        self.estimator = estimator
        self.param_grid = param_grid
        self.thresholds = thresholds
        self.scoring = scoring
        self.n_jobs = n_jobs
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


    def _handle_X_y(self, X, y=None):
        return _handle_X_y_dask(X, y=y)

    ####################################################################
    # SKLEARN / DASK GridSearchCV Methods ##############################

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

        # pizza, validate() was moved from after self._classes = None
        # on 24_07_26_19_32_00. any problems, move it back.
        self._validate_and_reset()

        _X, _y, _feature_names_in, _n_features_in = self._handle_X_y(X, y)


        # DONT unique().compute() HERE, JUST RETAIN THE VECTOR & ONLY DO
        # THE PROCESSING IF classes_ IS CALLED
        self._classes_ = y

        # THIS IS A HOLDER THAT IS FILLED ONE TIME WHEN THE unique().compute()
        # IS DONE ON self._classes_
        self._classes = None


        # BEFORE RUNNING cv_results_builder, THE THRESHOLDS MUST BE REMOVED FROM EACH PARAM GRID IN wip_param_grids
        # BUT THEY NEED TO BE RETAINED FOR CORE GRID SEARCH.
        THRESHOLD_DICT = {i:self.wip_param_grid[i].pop('thresholds') for i in range(len(self.wip_param_grid))}

        # pizza 24_07_10, for both sk and dask, n_splits_ is only
        # available after fit(). n_splits_ is always returned as a number
        self.cv_results_, PARAM_GRID_KEY = \
            _cv_results_builder(self.wip_param_grid, self.n_splits_, self.scorer_, self.return_train_score)



        if callable(self.wip_refit):
            _verify_refit_callable(self.wip_refit, deepcopy(self.cv_results_))

        with self._scheduler or distributed.Client(processes=False) as scheduler:
            self.cv_results_ = _core_fit(
                _X,
                _y,
                self._estimator,
                self.cv_results_,
                self._cv,
                self._error_score,
                self._verbose,
                self.scorer_,
                self._cache_cv,
                self._iid,
                self._return_train_score,
                PARAM_GRID_KEY,
                THRESHOLD_DICT,
                **params
            )

        super().fit(_X, _y, **params)

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
        return dask.visualize(self._estimator, filename=filename, format=format)


    # END SKLEARN / DASK GridSearchCV Method ###########################
    ####################################################################


    ####################################################################
    # SUPPORT METHODS ##################################################




    def _validate_and_reset(self):

        super()._validate_and_reset()

        _validate_dask_estimator(self.estimator)
        self._estimator = self.estimator

        self._iid = _validate_iid(self.iid)

        self._scheduler = _validate_scheduler(self.scheduler)

        self._cache_cv = _validate_cache_cv(self.cache_cv)

    # END SUPPORT METHODS ##############################################
    ####################################################################





















