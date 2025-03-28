# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Literal,
    Iterable,
    Sequence,
    Callable,
    Optional
)
from typing_extensions import Union
from .._type_aliases import (
    XInputType,
    YInputType,
)

import numbers
from copy import deepcopy

import distributed

from ...GSTCV._GSTCVMixin._GSTCVMixin import _GSTCVMixin

from ...GSTCV._GSTCVDask._validation._cache_cv import \
    _validate_cache_cv
from ...GSTCV._GSTCVDask._validation._scheduler import \
    _validate_scheduler
from ...GSTCV._GSTCVDask._validation._iid import _validate_iid
from ...GSTCV._GSTCVDask._validation._dask_estimator import \
    _validate_dask_estimator

from ...GSTCV._GSTCVDask._handle_X_y._handle_X_y_dask import \
    _handle_X_y_dask

from ...GSTCV._GSTCVDask._fit._core_fit import _core_fit

from ....base import check_is_fitted



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

    ********************************************************************

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
        Union[dict[str, Sequence[any], list[dict[str, Sequency[any]]]]] -
        Dictionary with keys as parameters names (str) and values as
        lists of parameter settings to try as values, or a list of such
        dictionaries. When multiple param grids are passed in a list,
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
        Optional[bool, str, Callable], default=True - After completion
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
        dict[str, np.ma.maskedarray] - A dictionary with column headers
        as keys and results as values, that can be conveniently converted
        into a pandas DataFrame.

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
        dict[str, any] - Exposes the dictionary found at
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


    def __init__(self,
        estimator,
        param_grid: Union[
            dict[str, Sequence[any]], Sequence[dict[str, Sequence[any]]]
        ],
        *,
        thresholds:
            Optional[Union[numbers.Real, Sequence[numbers.Real], None]]=None,
        scoring: Optional[
            Union[Sequence[str], dict[str, Callable], str, Callable]
        ]='accuracy',
        iid: Optional[bool]=True,
        refit: Optional[Union[bool, str, Callable, None]] = True,
        cv: Optional[Union[int, Iterable, None]]=None,
        verbose: Optional[numbers.Real]=0,
        error_score: Optional[Union[Literal['raise'], int, float]]='raise',
        return_train_score: Optional[bool]=False,
        scheduler: Optional[
            Union[distributed.Client, distributed.scheduler.Scheduler, None]
        ]=None,
        n_jobs: Optional[Union[int, None]]=None,
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


    ####################################################################
    # SUPPORT METHODS ##################################################

    def _handle_X_y(self, X, y=None):

        """

        Implements _handle_X_y_dask in methods in _GSTCVMixin.
        See the docs for _handle_X_y_dask.

        """

        return _handle_X_y_dask(X, y=y)


    def _core_fit(
            self,
            X: XInputType,
            y: YInputType=None,
            **params
        ):

        """

        GSTCVDask-specific function supporting fit(); called by
        _GSTCVMixin.fit()

        Perform all fit, scoring, and tabulation activities for every
        search performed in finding the hyperparameter values that
        maximize score (or minimize loss) for the given dataset (X)
        against the given target (y.)

        Returns all search results (times, scores, thresholds) in the
        cv_results dictionary.


        Parameters
        ----------
        X:
            dask.array.core.Array[Union[int, float]] - the data to be fit
            by GSTCVDask against the target.
        y:
            dask.array.core.Array[Union[int, float]] - the target to
            train the data against.


        Return
        ------
        -
            _cv_results: dict[str: np.ma.masked_array] - dictionary
                populated with all the times, scores, thresholds,
                parameter values, and search grids for every permutation
                of grid search.

        """


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

        'visualize' is not implemented in GSTCVDask.

        """


        check_is_fitted(self, attributes='_refit')

        __ = type(self).__name__
        raise NotImplementedError(
            f"visualize is not implemented in {__}."
        )


    def _validate_and_reset(self):

        """

        Perform initialization and validation for GSTCVDask-specific and
        shared parameters.

        """

        super()._validate_and_reset()

        _validate_dask_estimator(self.estimator)

        self._estimator = type(self.estimator)(
            **deepcopy(self.estimator.get_params(deep=False))
        )
        self._estimator.set_params(
            **deepcopy(self.estimator.get_params(deep=True))
        )

        self._iid = _validate_iid(self.iid)

        self._scheduler = _validate_scheduler(self.scheduler, self._n_jobs)

        self._cache_cv = _validate_cache_cv(self.cache_cv)

    # END SUPPORT METHODS ##############################################
    ####################################################################





















