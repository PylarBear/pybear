# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
from typing import Union, Literal, Iterable, Optional

import dask, distributed

from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    ScorerInputType,
    RefitType,
    ParamGridType,
    SchedulerType
)

from model_selection.GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from ._validation._cache_cv import _validate_cache_cv
from ._validation._scheduler import _validate_scheduler
from ._validation._iid import _validate_iid
from ._validation._dask_estimator import _validate_dask_estimator

from ._handle_X_y._handle_X_y_dask import _handle_X_y_dask


from model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit

# pizza! 24_07_14..... prevent empty param_grid from getting into _core_fit....
# this may be done already with a len(param_grid) but make sure.


class GSTCVDask(_GSTCVMixin):

    """

    24_08_11..... pizza remember that in pytest GSTCVDask was converting
    Dask KFold generator to empty list. Check this again when testing
    GSTCVDask.

    24_08_11..... pizza, when scheduler is not passed, and there is no
    global scheduler/client, multiprocesses Client is setup with n_jobs
    number of workers and 1 thread per worker. If n_jobs is None,
    the default behavior for distributed.Client when None is passed to
    n_workers is used.

    24_08_11...... pizza, iid is ignored when cv is an iterable.

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
        thresholds: Optional[Union[Iterable[Union[int, float]], int, float, None]]=None,
        scoring: Optional[ScorerInputType]='accuracy',
        iid: Optional[bool]=True,
        refit: Optional[RefitType]=True,
        cv: Optional[Union[int, None]]=None,
        verbose: Optional[Union[int, float, bool]]=0,
        error_score: Optional[Union[Literal['raise'], int, float]]='raise',
        return_train_score: Optional[bool]=False,
        scheduler: Optional[Union[SchedulerType, None]]=None,
        n_jobs: Optional[Union[int,None]]=None,
        cache_cv: Optional[bool]=True
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


    ####################################################################
    # SUPPORT METHODS ##################################################

    def _handle_X_y(self, X, y=None):

        # PIZZA PIZZA! WHAT ABOUT BLOCKING NON-[0,1] y? THINK ON OVR 🤔

        return _handle_X_y_dask(X, y=y)


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
            self._cache_cv,
            self._iid,
            self._return_train_score,
            self._PARAM_GRID_KEY,
            self._THRESHOLD_DICT,
            **params
        )

        return


    def visualize(self, filename="mydask", format=None, **kwargs):
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

        # if hasattr(self, '_dask_estimator') and not self._dask_estimator:
        #     raise NotImplementedError(f"Cannot visualize a sklearn estimator")
        #
        # self.check_is_fitted()
        #
        # return dask.visualize(
        #     self._estimator,
        #     filename=filename,
        #     format=format,
        #     **kwargs
        # )

        self.check_is_fitted()

        __ = type(self).__name__
        raise NotImplementedError(
            f"visualize is not implemented in {__}."
        )


    def _validate_and_reset(self):

        super()._validate_and_reset()

        _validate_dask_estimator(self.estimator)
        self._estimator = self.estimator

        self._iid = _validate_iid(self.iid)

        self._scheduler = _validate_scheduler(self.scheduler, self._n_jobs)

        self._cache_cv = _validate_cache_cv(self.cache_cv)

    # END SUPPORT METHODS ##############################################
    ####################################################################





















