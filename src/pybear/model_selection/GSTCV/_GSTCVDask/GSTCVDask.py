# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Literal,
    Iterable,
    Sequence,
    Optional
)
from typing_extensions import (
    Any,
    Union
)
from ._type_aliases import (
    DaskXType,
    DaskYType,
    DaskKFoldType,
    DaskSplitType,
    DaskSchedulerType
)
from .._type_aliases import (
    ClassifierProtocol,
    ParamGridInputType,
    ParamGridsInputType,
    ParamGridsWIPType,
    ThresholdsWIPType,
    MaskedHolderType,
    NDArrayHolderType
)

from copy import deepcopy
import numbers

from dask import compute
import distributed

from ._validation._validation import _validation
from ._validation._y import _val_y

from ._param_conditioning._scheduler import _cond_scheduler

from .._GSTCVDask._fit._get_kfold import _get_kfold as _dask_get_kfold
from .._GSTCVDask._fit._fold_splitter import _fold_splitter as _dask_fold_splitter
from .._GSTCVDask._fit._estimator_fit_params_helper import _estimator_fit_params_helper as _dask_estimator_fit_params_helper
from .._GSTCVDask._fit._parallelized_fit import _parallelized_fit
from .._GSTCVDask._fit._parallelized_scorer import _parallelized_scorer
from .._GSTCVDask._fit._parallelized_train_scorer import _parallelized_train_scorer
from .._GSTCVMixin._GSTCVMixin import _GSTCVMixin



class GSTCVDask(_GSTCVMixin):

    """

    Exhaustive cross-validated search over a grid of parameter values
    and decision thresholds for a binary classifier. The optimal
    parameters and decision threshold selected are those that maximize
    the score of the held-out data (test sets).

    GSTCVDask implements “fit”, “predict_proba”, "predict", “score”,
    "get_params", and "set_params" methods. It also implements
    “decision_function”, "predict_log_proba", “score_samples”,
    “transform” and “inverse_transform” if they are exposed by the
    classifier used.


    Parameters
    ----------
    estimator:
        estimator object - Must be a binary classifier that conforms to
        the sci-kit learn estimator API interface. The classifier must
        have 'fit', 'set_params', 'get_params', and 'predict_proba'
        methods. If the classifier does not have predict_proba, try to
        wrap with CalibratedClassifierCV. The classifier does not need a
        'score' method, as GSTCVDask never accesses the estimator score
        method because it always uses a 0.5 threshold.

        GSTCVDask warns when a non-dask estimator is used, but does not
        strictly prohibit them. GSTCVDask is explicitly designed for use
        with dask objects (estimators, arrays, and dataframes.) GSTCV is
        recommended for non-dask classifiers.

    param_grid:
        dict[str, Sequence[Any] or Sequence[dict[str, Sequency[Any]]] -
        Dictionary with keys as parameters names (str) and values as
        lists of parameter settings to try as values, or a list of such
        dictionaries. When multiple param grids are passed in a list-like,
        the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.

    thresholds:
        Optional[Union[None, Union[numbers.Real], Sequence[numbers.Real]]] -
        The decision threshold search grid to use when performing
        hyperparameter search. Other GridSearchCV modules only allow for
        search at the conventional decision threshold for binary class-
        ifiers, 0.5. This module additionally searches over any set of
        decision threshold values in the 0 to 1 interval (inclusive) in
        the same manner as any other hyperparameter while performing the
        grid search.

        The thresholds parameter can be passed via the 'thresholds'
        kwarg. In this case, thresholds can be None, a single number
        from 0 to 1 (inclusive) or a list-like of such numbers. If None,
        (and thresholds are not passed directly inside the param grid(s)),
        the default threshold grid is used, numpy.linspace(0, 1, 21).

        Thresholds may also be passed to individual param grids via a
        'thresholds' key. However, when passed directly to a param grid,
        thresholds cannot be None or a single number, it must be a list-
        like of numbers as is normally done with param grids.

        Because 'thresholds' can be passed in 2 different ways, there is
        a hierarchy that dictates which thresholds are used during
        searching and scoring. Any threshold values passed directly
        within a param grid always supersede any passed (or not passed)
        to the 'thresholds' kwarg. When no thresholds are passed inside
        a param grid, the values passed as a kwarg are used -- if no
        values were passed as a kwarg, then the default values are
        used. If all passed param grids have no 'thresholds' entry, then
        whatever is passed to the kwarg is used for all of them; if the
        'thresholds' kwarg is left as default, then the default threshold
        grid is used for all the grids.

        When one scorer is used, the best threshold is always exposed
        and is accessible via the best_threshold_ attribute. When
        multiple scorers are used, the best_threshold_ attribute is only
        exposed when a string value is passed to the refit kwarg.
        The best threshold is never reported in the best_params_
        attribute, even if thresholds were passed via a param grid; the
        best threshold is only available conditionally via the
        best_threshold_ attribute. Another way to discover the best
        threshold for each scorer is by inspection of the cv_results_
        attribute.

        The scores reported for test data in cv_results_ are those for
        the best threshold. Also note that when return_train_score is
        True, the scores returned for the train data are only for the
        best threshold found for the test data. That is, the thresholds
        are scored for the test data, the best score is found, and the
        best threshold is set based on the threshold for that score.
        Then when scoring train data, only that threshold is scored and
        reported in cv_results_.

    scoring:
        Optional[Union[str, Callable, Sequence[str], dict[str, Callable]]],
        default='accuracy' - Strategy to evaluate the performance of the
        cross-validated model on the test set (and also train set, if
        return_train_score is True.)

        For any number of scorers, scoring can be a dictionary with user-
        assigned scorer names as keys and callables as values. See below
        for clarification on allowed callables.

        For a single scoring metric, a single string or a single callable
        is also allowed. Valid strings that can be passed are 'accuracy',
        'balanced_accuracy', 'average_precision', 'f1', 'precision', and
        'recall'.

        For evaluating multiple metrics, scoring can also be a vector-
        like of unique strings, containing a combination of the allowed
        strings.

        The default scorer of the estimator cannot used by this module
        because the decision threshold cannot be manipulated. Therefore,
        'scoring' cannot accept a None argument.

        About the scorer callable:
        This module's scorers deviate from other GridSearch implement-
        ations in an important way. Some of those implementations accept
        make_scorer functions, e.g. sklearn.metrics.make_scorer, but
        this module cannot accept this. make_scorer implicitly assumes a
        decision threshold of 0.5, but this module needs to be able to
        calculate predictions based on any user-entered threshold. There-
        fore, in place of make_scorer functions, this module uses scoring
        metrics directly (whereas they would otherwise be passed to
        make_scorer.)

        Additionally, this module can accept any scoring function that
        has signature (y_true, y_pred) and returns a single number. Note
        that, when using a custom scorer, the scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        This module cannot directly accept scorer kwargs and pass them
        to scorers. To pass kwargs to your scoring metric, create a
        wrapper with signature (y_true, y_pred) around the metric and
        hard-code the kwargs into the metric, e.g.,

        def your_metric_wrapper(y_true, y_pred):
            return your_metric(y_true, y_pred, **hard_coded_kwargs)

    iid:
        Optional[bool], default=True - iid is ignored when cv is an
        iterable. Indicates whether the data's examples are believed to
        have random distribution (True) or if the examples are organized
        non-randomly in some way (False). If the data is not iid, dask
        KFold will cross chunk boundaries when reading the data in an
        attempt to randomize the data; this can be an expensive process.
        Otherwise, if the data is iid, dask KFold can handle the data as
        chunks which is much more efficient.

    refit:
        Optional[Union[bool, str, Callable]], default=True - After completion
        of the grid search, fit the estimator on the whole dataset using
        the best found parameters, and expose this fitted estimator via
        the best_estimator_ attribute. Also, when the estimator is refit
        the GSTCV instance itself becomes the best estimator, exposing
        the predict_proba, predict, and score methods (and possibly
        others.) When refit is not performed, the search simply finds
        the best parameters and exposes them via the best_params_
        attribute (unless there are multiple scorers and refit is False,
        in which case information about the grid search is only available
        via the cv_results_ attribute.)

        The values accepted by refit depend on the scoring scheme, that
        is, whether a single or multiple scorers are used. In all cases,
        refit can be boolean False (to disable refit), a string that
        indicates the scorer to use to determine the best parameters
        (when there is only one scorer there is only one possible string
        value), or a callable. See below for more information about the
        refit callable. When one metric is used, refit can be boolean
        True or False, but boolean True cannot be used when there is
        more than one scorer.

        Where there are considerations other than maximum score in
        choosing a best estimator, refit can be set to a function that
        takes in cv_results_ and returns the best_index_ (an integer).
        In that case, best_params_ and best_estimator_ will be set
        according to the returned best_index_. The best_score_ and
        best_threshold_ attributes will not be available if there are
        multiple scorers, but are available if there is only one scorer.

        See scoring parameter to know more about multiple metric
        evaluation.

    cv:
        Optional[Union[int, Sequence, None]], default=None - Sets the
        cross-validation splitting strategy.

        Possible inputs for cv are:
        1) None, to use the default 5-fold cross validation,
        2) integer, must be 2 or greater, to specify the number of folds
            in a (Stratified)KFold,
        3) An iterable yielding pairs of (train, test) split indices as
            arrays.

        For passed iterables:
        This module will convert generators to lists. No validation is
        done beyond verifying that it is an iterable that contains pairs
        of iterables. GSTCV will catch out of range indices and raise
        but any validation beyond that is up to the user outside of
        GSTCV.

    verbose:
        Optional[numbers.Real], default=0 - The amount of verbosity to
        display to screen during the grid search. Accepts integers from
        0 to 10. 0 means no information displayed to the screen, 10
        means full verbosity. Non-numbers are rejected. Boolean False is
        set to 0, boolean True is set to 10. Negative numbers are
        rejected. Numbers greater than 10 are set to 10. Floats are
        rounded to integers.

    error_score:
        Optional[Union[int, float, Literal['raise']]], default='raise' -
        Score to assign if an error occurs in estimator fitting. If set
        to ‘raise’, the error is raised. If a numeric value is given, a
        warning is raised and the error score value is inserted into the
        subsequent calculations in place of the missing value(s). This
        parameter does not affect the refit step, which will always raise
        the error.

    return_train_score:
        Optional[bool] - If False, the cv_results_ attribute will not
        include training scores. Computing training scores is used to
        get insights on how different parameter settings impact the
        overfitting/underfitting trade-off. However, computing the scores
        on the training set can be computationally expensive and is not
        strictly required to select the parameters that yield the best
        generalization performance.

    scheduler:
        Optional[Union[distributed.Client, distributed.scheduler.Scheduler,
        None]], default=None -

        A passed scheduler supersedes all other external schedulers.
        When a scheduler is explicitly passed, GSTCVDask does not perform
        any validation or verification but allows that to be handled by
        dask at compute time.

        If "None" was passed to the scheduler kwarg (the default),
        GSTCVDask looks for an external context manager or global
        scheduler using get_client. If one exists, GSTCVDask uses that
        as the scheduler. If an external scheduler does not exist,
        GSTCVDask instantiates a multiprocessing distributed.Client()
        (which defaults to LocalCluster) with n_workers=n_jobs and 1
        thread per worker. If n_jobs is None, GSTCVDask uses the default
        distributed.Client behavior when n_workers is set to None.

        This module intentionally disallows any shorthand methods for
        internally setting up a scheduler (such as strings like 'thread-
        ing' and 'multiprocessing', which are ultimately passed to
        dask.base.get_scheduler.) All of these types of configurations
        should be handled by the user external to the GSTCVDask module.
        As much as possible, dask and distributed objects are allowed to
        flow through without any hard-coded input.

    n_jobs:
        Optional[Union[int, None]], default=None - Active only if no
        scheduler is available. That is, if a scheduler is not passed to
        the scheduler kwarg, if no global scheduler is available, and if
        there is no scheduler context manager, only then does n_jobs
        become effectual. In this case, GSTCVDask creates a distributed
        Client multiprocessing instance with n_workers=n_jobs.

    cache_cv:
        Optional[bool], default=True - Indicates if the train/test folds
        are to be stored when first generated, or if the folds are
        generated from X and y with the KFold indices at each point of
        use.

    ********************************************************************

    Attributes
    ----------
    cv_results_:
        CVResultsType - A dictionary with column headers as keys and
        results as values, that can be conveniently converted into a
        pandas DataFrame.

        Always exposed after fit.

        Below is an example of cv_results_ for a logistic classifier,
        with:
            cv=3,
            param_grid={'C': [1e-5, 1e-4]},
            thresholds=np.linspace(0,1,21),
            scoring=['accuracy', 'balanced_accuracy']
            return_train_score=False

        on random data.

        {
            'mean_fit_time':                    [1.227847, 0.341168]
            'std_fit_time':                     [0.374309, 0.445982]
            'mean_score_time':                  [0.001638, 0.001676]
            'std_score_time':                   [0.000551, 0.000647]
            'param_C':                             [0.00001, 0.0001]
            'params':                  [{'C': 1e-05}, {'C': 0.0001}]
            'best_threshold_accuracy':                   [0.5, 0.51]
            'split0_test_accuracy':              [0.785243, 0.79844]
            'split1_test_accuracy':              [0.80228, 0.814281]
            'split2_test_accuracy':             [0.805881, 0.813381]
            'mean_test_accuracy':               [0.797801, 0.808701]
            'std_test_accuracy':                [0.009001, 0.007265]
            'rank_test_accuracy':                             [2, 1]
            'best_threshold_balanced_accuracy':          [0.5, 0.51]
            'split0_test_balanced_accuracy':    [0.785164, 0.798407]
            'split1_test_balanced_accuracy':    [0.802188, 0.814252]
            'split2_test_balanced_accuracy':    [0.805791, 0.813341]
            'mean_test_balanced_accuracy':      [0.797714, 0.808667]
            'std_test_balanced_accuracy':       [0.008995, 0.007264]
            'rank_test_balanced_accuracy':                    [2, 1]
        }

        Slicing across the masked arrays yields the results for the fit
        and score of a single set of search points. That is, indexing
        into all of the masked arrays at position zero yields the result
        for the first set of search points, index 1 contains the results
        for the second set of points, and so forth.

        The key 'params' is used to store a list of parameter settings
        dicts for all the parameter candidates. That is, the 'params'
        key holds all the possible permutations of parameters for the
        given search grid(s).

        The mean_fit_time, std_fit_time, mean_score_time and
        std_score_time are all in seconds.

        For single-metric evaluation, the scores for the single scorer
        are available in the cv_results_ dict at the keys ending with
        '_score'. For multi-metric evaluation, the scores for all the
        scorers are available in the cv_results_ dict at the keys ending
        with that scorer’s name ('_<scorer_name>').
        (‘split0_test_precision’, ‘mean_train_precision’ etc.)

    best_estimator_:
        estimator - The estimator that was chosen by the search, i.e.
        the estimator which gave highest score (or smallest loss if
        specified) on the held-out (test) data. Only exposed when refit
        is not False; see refit parameter for more information on allowed
        values.

    best_score_:
        float - The mean of the scores of the hold out (test) cv folds
        for the best estimator. Always exposed when there is one scorer,
        or when refit is specified as a string for 2+ scorers.

    best_params_:
        dict[str, Any] - Exposes the dictionary found at
        cv_results_['params'][best_index_], which gives the parameter
        settings that resulted in the highest mean score (best_score_)
        on the hold out (test) data.

        best_params_ never holds best_threshold_. Access best_threshold_
        via the best_threshold_ attribute (if available) or the
        cv_results_ attribute.

        best_params_ is always exposed when there is one scorer, or when
        refit is not False for 2+ scorers.

    best_index_:
        int - The index of the cv_results_ arrays which corresponds to
        the best parameter settings. Always exposed when there is one
        scorer, or when refit is not False for 2+ scorers.

    scorer_:
        dict - Scorer metric(s) used on the held out data to choose the
        best parameters for the model. Always exposed after fit.

        This attribute holds the validated scoring dictionary which maps
        the scorer key to the scorer metric callable, i.e., a dictionary
        of {scorer_name: scorer_metric}.

    n_splits_:
        int -  The number of cross-validation splits (folds/iterations).
        Always exposed after fit.

    refit_time_:
        float - Seconds elapsed when refitting the best model on the
        whole dataset. Only exposed when refit is not False.

    multimetric_:
        bool - Whether or not several scoring metrics were used. False
        if one scorer was used, otherwise True. Always exposed after fit.

    classes_:
        ndarray of shape (n_classes,) - Class labels. Only exposed when
        refit is not False. Because GSTCV imposes a restriction that y
        must be binary in [0, 1], this must always return [0, 1].

    feature_names_in_:
        ndarray of shape (n_features_in_,) - Names of features seen
        during fit.

        Only exposed when refit is not False (see the documentation for
        the refit parameter for more details) and a dataframe was passed
        to fit.

    best_threshold_:
        float - The threshold that, when used along with the parameter
        values found in best_params_, yields the highest score for the
        given settings and data.

        When one scorer is used, the best threshold found is always
        exposed via the best_threshold_ attribute. When multiple scorers
        are used, the best_threshold_ attribute is only exposed when a
        string value is passed to the refit kwarg.

        The best threshold is only available conditionally via the
        best_threshold_ attribute. Another way to discover the best
        threshold for each scorer is by inspection of the cv_results_
        attribute.

    ********************************************************************

    Notes
    -----

    Validation of X and y passed to GSTCVDask methods always checks for
    at least 2 things. First, both X and y must be numeric (i.e., can
    pass a test where they are converted to np.uint8.) GSTCVDask (and
    most estimators) cannot accept non-numeric data. Secondly, y must be
    a single label and binary in 0,1.

    Conditionally, when fit was done on a dataframe, a dataframe passed
    to a method is checked for exact matching of the names and order of
    the columns.

    Any other exception that is raised by a GSTCVDask method besides
    these 3 is being raised from within the estimator itself.

    ********************************************************************

    Examples
    --------
    >>> import numpy as np
    >>> from pybear.model_selection import GSTCVDask
    >>> from dask_ml.linear_model import LogisticRegression
    >>> from dask_ml.datasets import make_classification

    >>> clf = LogisticRegression(
    ...     solver='lbfgs',
    ...     tol=1e-4
    ... )
    >>> X, y = make_classification(
    ...     n_samples=90, n_features=5, random_state=19, chunks=(90, 5)
    ... )
    >>> param_grid = {
    ...     'C': [1e-4, 1e-3],
    ... }
    >>> gstcv = GSTCVDask(
    ...     estimator=clf,
    ...     param_grid=param_grid,
    ...     thresholds=[0.1, 0.5, 0.9],
    ...     scoring='balanced_accuracy',
    ...     iid=True,
    ...     refit=False,
    ...     cv=3,
    ...     verbose=0,
    ...     error_score='raise',
    ...     return_train_score=False,
    ...     scheduler=None,
    ...     n_jobs=None,
    ...     cache_cv=True
    ... )
    >>> gstcv.fit(X, y)
    GSTCVDask(cv=3, estimator=LogisticRegression(solver='lbfgs'),
              param_grid={'C': array([0.0001, 0.001 ])}, refit=False,
              scoring='balanced_accuracy', thresholds=[0.1, 0.5, 0.9])

    >>> gstcv.best_params_
    {'C': 0.0001}

    >>> gstcv.best_threshold_
    0.5

    """


    def __init__(
        self,
        estimator: ClassifierProtocol,
        param_grid: Union[ParamGridInputType, ParamGridsInputType],
        *,
        thresholds: Optional[Union[None, numbers.Real, Sequence[numbers.Real]]]=None,
        scoring: Optional[
            Union[str, Sequence[str], Callable, dict[str, Callable]]
        ]='accuracy',
        iid: Optional[bool]=True,
        refit: Optional[Union[bool, str, Callable]] = True,
        cv: Optional[Union[numbers.Integral, Iterable, None]]=None,
        verbose: Optional[numbers.Real]=0,
        error_score: Optional[Union[Literal['raise'], numbers.Real]]='raise',
        return_train_score: Optional[bool]=False,
        scheduler: Optional[
            Union[distributed.Client, distributed.scheduler.Scheduler, None]
        ]=None,
        n_jobs: Optional[Union[numbers.Integral, None]]=None,
        cache_cv: Optional[bool]=True
    ) -> None:

        """Initialize the GSTCVDask instance."""

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


    def _val_y(self, _y: DaskYType) -> None:

        """
        Implements GSTCVDask _val_y in methods in _GSTCVMixin. See the
        docs for GSTCVDask _val_y.


        Parameters
        ----------
        _y:
            DaskYType - the target for the data.


        Returns
        -------
        -
            None

        """

        # KEEP val of X & y separate, the all methods need X & y everytime
        _val_y(_y)


    def _val_params(self) -> None:

        """
        Validate init params that are unique to GSTCVDask.


        Returns
        -------
        -
            None

        """

        # KEEP val of y separate
        _validation(self.estimator, self.iid, self.cache_cv)


    def _condition_params(
        self,
        _X: DaskXType,
        _y: DaskYType
    ) -> None:

        """
        Condition GSTCVDask-only init params into format for internal
        processing.


        Parameters
        ----------
        _X:
            DaskXType: The data.
        _y:
            DaskYType: The target for the data.


        Returns
        -------
        -
            None

        """


        self._scheduler: DaskSchedulerType = \
            _cond_scheduler(self.scheduler, self.n_jobs)

        self._KFOLD: list[DaskKFoldType]
        if isinstance(self._cv, numbers.Integral):
            self._KFOLD = list(_dask_get_kfold(
                _X, self.n_splits_, self.iid, self._verbose, _y=_y
            ))
        else:  # _cv is an iterable, _cond_cv should have made list[tuple]
            self._KFOLD = self._cv

        self._CACHE_CV: Union[None, list[DaskSplitType]] = None
        if self.cache_cv:
            self._CACHE_CV = []
            for (train_idxs, test_idxs) in self._KFOLD:
                self._CACHE_CV.append(_dask_fold_splitter(train_idxs, test_idxs, _X, _y))
        # pizza at some point think on scattering _CACHE_CV
        # deal with the persists in the parallelized files


    def _fit_all_folds(
        self,
        _X:DaskXType,
        _y:DaskYType,
        _grid:dict[str, Any],
        _fit_params
    ) -> list[tuple[ClassifierProtocol, float, bool], ...]:

        """
        Fit on each train/test split for one single set of hyperparameter
        values (one permutation of GSCV).


        Parameters
        ----------
        _X:
            DaskXType - the data.
        _y:
            DaskYType - the target for the data.
        _grid:
            dict[str, Any] - the values for the hyperparameters for this
            permutation of grid search.


        Returns
        -------
        -
            list[tuple[ClassifierProtocol, float, bool], ...] - a list
            of tuples, one tuple for each fold, with each tuple holding
            the respective fitted estimator for that fold of train/test
            data, the fit time, and a bool indicating whether the fit
            raised an error.

        """

        # **** IMPORTANT NOTES ABOUT estimator & n_jobs ****
        # this is from sklearn notes, where the problem was discovered
        # there was a (sometimes large, >> 0.10) anomaly in scores
        # when n_jobs was set to (1, None) vs. (-1, 2, 3, 4) when using
        # SK Logistic. While running the fits under a regular for-loop,
        # the SK estimator itself was found to be the cause, from being
        # fit on repeatedly without reconstruct. There appears be some
        # form of state information being retained in the estimator that
        # is altered with fit and alters subsequent fits slightly. there
        # is very little on google and ChatGPT about this (ChatGPT also
        # thinks that 'state' is the problem.) Locking random_state made
        # no difference, and locking other things with randomness (KFold,
        # etc.) made no difference. There was no experimentation done to
        # see if that problem extends beyond SK Logistic or SK. joblib
        # with n_jobs (1, None) behaves exactly the same as a regular
        # for-loop. The solution is to reconstruct the estimator with each
        # fit and fit it once - then the results agree exactly with the
        # results when n_jobs > 1 and with SK gscv. 'estimator' is being
        # rebuilt at each call with the class constructor
        # type(_estimator)(**_estimator.get_params(deep=True)).

        # must use shallow params to construct estimator
        s_p = self._estimator.get_params(deep=False)  # shallow_params
        # must use deep params for pipeline to set GSCV params (depth
        # doesnt matter for an estimator when not pipeline.)
        d_p = self._estimator.get_params(deep=True)  # deep_params

        _fold_fit_params = _dask_estimator_fit_params_helper(
            [*compute(len(_y))][0],
            _fit_params,
            self._KFOLD
        )

        FIT_OUTPUT = []
        for f_idx in range(self.n_splits_):

            # train only!
            if self.cache_cv:
                _X_y_train = list(zip(*self._CACHE_CV[f_idx]))[0]
            elif not self.cache_cv:
                _X_y_train = list(zip(*_dask_fold_splitter(*self._KFOLD[f_idx], _X, _y)))[0]

            FIT_OUTPUT.append(
                _parallelized_fit(
                    f_idx,
                    *_X_y_train,
                    type(self._estimator)(**s_p).set_params(**deepcopy(d_p)),
                    _grid,
                    self.error_score,
                    **_fold_fit_params[f_idx]
                )
            )

        del s_p, d_p, _fold_fit_params, _X_y_train

        return FIT_OUTPUT

        # END FIT ALL FOLDS ###############################################


    def _score_all_folds_and_thresholds(
        self,
        _X:DaskXType,
        _y:DaskYType,
        _FIT_OUTPUT:list[tuple[ClassifierProtocol, float, bool], ...],
        _THRESHOLDS:ThresholdsWIPType
    ) -> list[tuple[MaskedHolderType, MaskedHolderType], ...]:

        """
        For each fitted estimator associated with each fold, produce the
        y_pred vector for that fold's test data and score it against the
        actual y.


        Parameters
        ----------
        _X:
            DaskXType - the data.
        _y:
            DaskYType - the target for the data.
        _FIT_OUTPUT:
            list[tuple[ClassifierProtocol, float, bool], ...] - a list
            of tuples, one tuple for each fold, with each tuple holding
            the respective fitted estimator for that fold of train/test
            data, the fit time, and a bool indicating whether the fit
            raised an error.
        _THRESHOLDS:
            ThresholdsWIPType - the thresholds for which to calculate
            scores.


        Returns
        -------
        -
            list[tuple[MaskedHolderType, MaskedHolderType], ...] -
            TEST_THRESHOLD_x_SCORER__SCORE_LAYER:
                MaskedHolderType - masked array of shape (n_thresholds,
                n_scorers) holding the scores for each scorer over all of
                the thresholds.
            TEST_THRESHOLD_x_SCORER__SCORE_TIME_LAYER:
                MaskedHolderType - masked array of shape (n_thresholds,
                n_scorers) holding the times to score each scorer over
                all of the thresholds. .... pizza check this is it an average

        """

        TEST_SCORER_OUT = []
        for f_idx in range(self.n_splits_):

            # test only!
            if self.cache_cv:
                _X_y_test = list(zip(*self._CACHE_CV[f_idx]))[1]
            elif not self.cache_cv:
                _X_y_test = list(zip(*_dask_fold_splitter(*self._KFOLD[f_idx], _X, _y)))[1]

            TEST_SCORER_OUT.append(
                _parallelized_scorer(
                    *_X_y_test,
                    _FIT_OUTPUT[f_idx],
                    f_idx,
                    self.scorer_,
                    _THRESHOLDS,
                    self.error_score,
                    self._verbose
                )
            )

        del _X_y_test

        return TEST_SCORER_OUT
        # END SCORE ALL FOLDS & THRESHOLDS #################################


    def _score_train(
        self,
        _X:DaskXType,
        _y:DaskYType,
        _FIT_OUTPUT:list[tuple[ClassifierProtocol, float, bool], ...],
        _BEST_THRESHOLDS_BY_SCORER:NDArrayHolderType
    ) -> list[MaskedHolderType]:
        # TRAIN_SCORER_OUT is TRAIN_SCORER__SCORE_LAYER

        """
        Using the fitted estimator for each fold, all the scorers, and
        the best thresholds for each scorer, score the train data for
        each fold using all the scorers and the single best threshold
        for the respective scorer.


        Parameters
        ----------
        _X:
            DaskXType - the data.
        _y:
            DaskYType - the target for the data.
        _FIT_OUTPUT:
            list[tuple[ClassifierProtocol, float, bool], ...] - a list
            of tuples, one tuple for each fold, with each tuple holding
            the respective fitted estimator for that fold of train/test
            data, the fit time, and a bool indicating whether the fit
            raised an error.
        _BEST_THRESHOLDS_BY_SCORER:
            NDArrayHolderType - the best thresholds found for each
            scorer as found by averaging the best thresholds across each
            fold of test data for each scorer ---- pizza verify this!


        Returns
        -------
        -
            list[MaskedHolderType] - list of masked arrays where
            each masked array holds the scores for a fold of train data
            using every scorer and the best threshold associated with
            that scorer.

        """

        TRAIN_SCORER_OUT = []
        for f_idx in range(self.n_splits_):

            # train only!
            if self.cache_cv:
                _X_y_train = list(zip(*self._CACHE_CV[f_idx]))[0]
            elif not self.cache_cv:
                _X_y_train = list(zip(*_dask_fold_splitter(*self._KFOLD[f_idx], _X, _y)))[0]

            TRAIN_SCORER_OUT.append(
                _parallelized_train_scorer(
                    *_X_y_train,
                    _FIT_OUTPUT[f_idx],
                    f_idx,
                    self.scorer_,
                    _BEST_THRESHOLDS_BY_SCORER,
                    self.error_score,
                    self._verbose
                )
            )

        del _X_y_train

        return TRAIN_SCORER_OUT





