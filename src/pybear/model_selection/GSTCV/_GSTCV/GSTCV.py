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

from .._type_aliases import ClassifierProtocol

from copy import deepcopy
import numbers

from ._validation._validation import _validation
from ._validation._X_y import _val_X_y

from ._fit._core_fit import _core_fit

from .._GSTCVMixin._GSTCVMixin import _GSTCVMixin



class GSTCV(_GSTCVMixin):

    """

    Exhaustive cross-validated search over a grid of parameter values
    and decision thresholds for a binary classifier. The optimal
    parameters and decision threshold selected are those that maximize
    the score of the held-out data (test sets).

    GSTCV implements “fit”, “predict_proba”, "predict", “score”,
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
        'score' method, as GSTCV never accesses the estimator score
        method because it always uses a 0.5 threshold.

        GSTCV deliberately blocks dask classifiers (including, but not
        limited to, dask_ml, xgboost, and lightGBM dask classifiers.) To
        use dask classifiers, use GSTCVDask.

    param_grid:
        dict[str, list-like] or list[dict[str, list-like]] - Dictionary
        with keys as parameters names (str) and values as lists of para-
        meter settings to try as values, or a list of such dictionaries.
        When multiple param grids are passed in a list, the grids spanned
        by each dictionary in the list are explored. This enables
        searching over any sequence of parameter settings.

    thresholds:
        Optional[Union[None, numbers.Real, Sequence[numbers.Real]] -
        The decision threshold search grid to use when performing hyper-
        parameter search. Other GridSearchCV modules only allow for
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

    n_jobs:
        Optional[Union[numbers.Integral, None]], default=None - Number
        of jobs to run in parallel. -1 means using all processors.

        For best speed benefit, pybear recommends setting n_jobs in both
        GSTCV and the wrapped estimator to None, whether under a joblib
        context manager or standing alone. When under a joblib context
        manager, also set n_jobs in the context manager to None.

    refit:
        Optional[Union[bool, str, Callable]], default=True - After
        completion of the grid search, fit the estimator on the whole
        dataset using the best found parameters, and expose this fitted
        estimator via the best_estimator_ attribute. Also, when the
        estimator is refit the GSTCV instance itself becomes the best
        estimator, exposing the predict_proba, predict, and score methods
        (and possibly others.) When refit is not performed, the search
        simply finds the best parameters and exposes them via the
        best_params_ attribute (unless there are multiple scorers and
        refit is False, in which case information about the grid search
        is only available via the cv_results_ attribute.)

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
        Optional[Union[numbers.Integral, Iterable, None]], default=None -
        Sets the cross-validation splitting strategy.

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
        0 to 10. 0 means no information displayed to the screen, 10 means
        full verbosity. Non-numbers are rejected. Boolean False is set
        to 0, boolean True is set to 10. Negative numbers are rejected.
        Numbers greater than 10 are set to 10. Floats are rounded to
        integers.

    pre_dispatch:
        Optional[Union[Literal['all'], str, numbers.Integral]],
        default='2*n_jobs' - The number of batches (of tasks) to be
        pre-dispatched. Default is '2*n_jobs'. See the joblib.Parallel
        docs for more information.

    error_score:
        Optional[Union[Literal['raise'], numbers.Real]], default='raise' -
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


    Notes
    -----

    Validation of X and y passed to GSTCV methods always checks for at
    least 2 things. First, both X and y must be numeric (i.e., can pass
    a test where they are converted to np.uint8.) GSTCV (and most
    estimators) cannot accept non-numeric data. Secondly, y must be a
    single label and binary in 0,1.

    Conditionally, when fit was done on a dataframe, a dataframe passed
    to a method is checked for exact matching of the names and order of
    the columns.

    Any other exception that is raised by a GSTCV method besides these 3
    is being raised from within the estimator itself.


    Examples
    --------
    >>> import numpy as np
    >>> from pybear.model_selection import GSTCV
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification

    >>> clf = LogisticRegression(
    ...     solver='saga',
    ...     penalty='elasticnet'
    ... )
    >>> X, y = make_classification(n_samples=1000, n_features=5, random_state=19)
    >>> param_grid = {
    ...     'C': np.logspace(-6, -3, 4),
    ...     'l1_ratio': np.linspace(0, 1, 5)
    ... }
    >>> gstcv = GSTCV(
    ...     estimator=clf,
    ...     param_grid=param_grid,
    ...     thresholds=np.linspace(0, 1, 5),
    ...     scoring='balanced_accuracy',
    ...     n_jobs=-1,
    ...     refit=False,
    ...     cv=5,
    ...     verbose=0,
    ...     pre_dispatch='2*n_jobs',
    ...     error_score='raise',
    ...     return_train_score=False
    ... )
    >>> gstcv.fit(X, y)
    GSTCV(cv=5, estimator=LogisticRegression(penalty='elasticnet', solver='saga'),
          n_jobs=-1,
          param_grid={'C': array([1.e-06, 1.e-05, 1.e-04, 1.e-03]),
                      'l1_ratio': array([0.  , 0.25, 0.5 , 0.75, 1.  ])},
          refit=False, scoring='balanced_accuracy',
          thresholds=array([0.  , 0.25, 0.5 , 0.75, 1.  ]))

    >>> gstcv.best_params_
    {'C': 0.001, 'l1_ratio': 0.25}

    >>> gstcv.best_threshold_
    0.5

    """


    def __init__(
        self,
        estimator: ClassifierProtocol,
        param_grid: Union[dict[str, Sequence[Any]], list[dict[str, Sequence[Any]]]],
        *,
        thresholds: Optional[Union[None, numbers.Real, Sequence[numbers.Real]]]=None,
        scoring: Optional[
            Union[list[str], dict[str, Callable], str, Callable]
        ]='accuracy',
        n_jobs: Optional[Union[numbers.Integral, None]]=None,
        refit: Optional[Union[bool, str, Callable]]=True,
        cv: Optional[Union[numbers.Integral, Iterable, None]]=None,
        verbose: Optional[numbers.Real]=0,
        pre_dispatch: Optional[
            Union[Literal['all'], str, numbers.Integral]
        ]='2*n_jobs',
        error_score: Optional[Union[Literal['raise'], numbers.Real]]='raise',
        return_train_score: Optional[bool]=False
    ) -> None:

        """Initialize the GSTCV instance."""

        self.estimator = estimator
        self.param_grid = param_grid
        self.thresholds = thresholds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score


    ####################################################################
    # SUPPORT METHODS ##################################################

    def _val_X_y(self, X, y:Optional[Any] = None):

        """
        Implements GSTCV _val_X_y in methods in _GSTCVMixin.
        See the docs for GSTCV _val_X_y.

        """

        return _val_X_y(X, _y=y)


    def _core_fit(
        self,
        X,    # pizza type hints here
        y=None,  # pizza type hints here
        **params
    ) -> None:

        """

        GSTCV-specific function supporting fit(); called by
        _GSTCVMixin.fit()

        Perform all fit, scoring, and tabulation activities for every
        search performed in finding the hyperparameter values that
        maximize score (or minimize loss) for the given dataset (X)
        against the given target (y.)

        Returns all search results (times, scores, thresholds) in the
        cv_results dictionary.


        Parameters
        ----------
        # pizza fix these wack type hints
        X:
            NDArray[Union[int, float]] - the data to be fit by GSTCV
            against the target.
        y:
            NDArray[Union[int, float]] - the target to train the data
            against.


        Return
        ------
        -
            _cv_results: dict[str: np.ma.masked_array] - dictionary
            populated with all the times, scores, thresholds, parameter
            values, and search grids for every permutation of grid
            search.

        """

        _validation(self.estimator, self.pre_dispatch)

        self.cv_results_ = _core_fit(
            X,
            y,
            self._estimator,
            self.cv_results_,
            self._cv,
            self.error_score,
            self._verbose,
            self.scorer_,
            self.n_jobs,
            self.pre_dispatch,
            self.return_train_score,
            self._PARAM_GRID_KEY,
            self._THRESHOLD_DICT,
            **params
        )

        return


    # END SUPPORT METHODS ##############################################
    ####################################################################







