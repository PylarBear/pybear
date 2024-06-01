# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
from typing import Union, TypeAlias, Iterable
import numpy as np


from ._autogridsearch_wrapper._type_aliases import ParamsType, BestParamsType

from ._autogridsearch_wrapper._print_results import _print_results

from ._autogridsearch_wrapper._validation._agscv_verbose import \
    _agscv_verbose as val_agscv_verbose
from ._autogridsearch_wrapper._validation._estimator import _estimator \
    as val_estimator
from ._autogridsearch_wrapper._validation._parent_gscv_kwargs import \
    _val_parent_gscv_kwargs
from ._autogridsearch_wrapper._validation._max_shifts import _max_shifts \
    as val_max_shifts
from ._autogridsearch_wrapper._validation._params__total_passes import \
    _params__total_passes as val_params_total_passes
from ._autogridsearch_wrapper._validation._total_passes_is_hard import \
    _total_passes_is_hard as val_total_passes_is_hard
from ._autogridsearch_wrapper._build_first_grid_from_params import _build
from ._autogridsearch_wrapper._build_is_logspace import _build_is_logspace

from ._autogridsearch_wrapper._get_next_param_grid._get_next_param_grid \
    import _get_next_param_grid
from ._autogridsearch_wrapper._demo._demo import _demo

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV as SklearnGridSearchCV,
    RandomizedSearchCV as SklearnRandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)

from dask_ml.model_selection import (
    GridSearchCV as DaskGridSearchCV,
    RandomizedSearchCV as DaskRandomizedSearchCV,
    IncrementalSearchCV,
    HyperbandSearchCV,
    SuccessiveHalvingSearchCV,
    InverseDecaySearchCV
)

from ..GSTCV.GSTCV import GridSearchThresholdCV





GridSearchType: TypeAlias = Union[
    type(SklearnGridSearchCV),
    type(SklearnRandomizedSearchCV),
    type(HalvingGridSearchCV),
    type(HalvingRandomSearchCV),
    type(DaskGridSearchCV),
    type(DaskRandomizedSearchCV),
    type(IncrementalSearchCV),
    type(HyperbandSearchCV),
    type(SuccessiveHalvingSearchCV),
    type(InverseDecaySearchCV),
    type(GridSearchThresholdCV)
]





def autogridsearch_wrapper(GridSearchParent: GridSearchType) -> GridSearchType:

    """
    Wrap a sci-kit learn or dask GridSearchCV class with a class that
    overwrites the fit method of GridSearchCV. The supersedent fit
    method automates multiple calls to the super fit() method using
    progressively narrower search grids based on previous search results.
    All sci-kit and dask GridSearch modules are supported. See the
    sci-kit learn and dask documentation for more information about the
    available GridSearchCV modules.

    Parameters
    ----------
    GridSearchParent:
        Sci-kit or Dask GridSearchCV class, not instance.

    Return
    ------
    -
        AutoGridSearch:
            Wrapped GridSearchCV class. The original fit method is
            replaced with a new fit method that can make multiple calls
            to the original fit method with increasingly convergent
            search grids.

    See Also
    --------
    sklearn.model_selection.GridSearchCV
    sklearn.model_selection.RandomizedSearchCV
    dask_ml.model_selection.HalvingGridSearchCV
    dask_ml.model_selection.HalvingRandomSearchCV
    dask_ml.model_selection.GridSearchCV
    dask_ml.model_selection.RandomizedSearchCV
    dask_ml.model_selection.IncrementalSearchCV
    dask_ml.model_selection.HyperbandSearchCV
    dask_ml.model_selection.SuccessiveHalvingSearchCV
    dask_ml.model_selection.InverseDecaySearchCV
    pybear.model_selection.GridSearchThresholdCV


    """


    class AutoGridSearch(GridSearchParent):


        def __init__(
            self,
            estimator,
            params: ParamsType,
            *,
            total_passes: int=5,
            total_passes_is_hard: bool=False,
            max_shifts: Union[None, int]=None,
            agscv_verbose: bool=False,
            **parent_gscv_kwargs
        ) -> None:

            """
            Run multiple passes of GridSearchCV with progressively narrower
            search spaces to find the most precise estimate of the best
            value for each hyperparameter. 'Best' values are those values
            within the given search grid that minimize loss or maximize
            score for the particular estimator and data set.

            Sklearn / dask GridSearchCV.best_params_ is a dictionary
            with parameter names as keys and respective best values
            as values. The fit() method of this wrapper class uses the
            best_params_ dictionary to calculate refined grids for the
            next search round and passes the grids to another dictionary
            that satisfies the 'param_grid' argument of GridSearchCV (or
            a different argument such as 'parameters' for other GridSearch
            modules.)

            On the first pass, search grids are constructed as instructed
            in the 'params' argument. On subsequent passes, calculated
            grids are constructed based on:
                • the preceding search grid,
                • the results within 'best_params_',
                • the parameters' datatypes as specified in 'params', and
                • the number of points as specified in 'params'.

            The new refined param_grid is passed to another round of
            GridSearchCV, best_params_ is retrieved, and another
            param_grid is created. This process is repeated at least
            total_passes number of times, with each successive pass
            returning increasingly precise estimates of the true best
            hyperparameter values for the given estimator, data set, and
            restrictions imposed in the params parameter.

            Example param_grid:
                {
                'C': [0,5,10],
                'l1_ratio': [0, 0.5, 1],
                'solver': ['lbfgs', 'saga']
                }

            Example best_params:
                {
                'C': 10,
                'l1_ratio': 0.5,
                'solver': 'lbfgs']
                }

            AutoGridSearch leaves the API of the parent GridSearchCV
            module intact, and all of the parent module's methods and
            attributes are accessible via the AutoGridSearch instance.
            This, however, precludes AutoGridSearch from using the same
            API itself (doing so would overwrite the underlying's) so
            methods like set_params, get_params, etc., access the parent
            GridSearchCV and not the child AutoGridSearch wrapper. The
            attributes of an AutoGridSearch instance (max_shifts,
            total_passes, etc.) can be accessed and set directly:

            >>> from pybear.model_selection import autogridsearch_wrapper
            >>> from sklearn.model_selection import GridSearchCV
            >>> from sklearn.linear_model import LogisticRegressions
            >>> AutoGSCV = autogridsearch_wrapper(GridSearchCV)
            >>> estimator = LogisticRegression()
            >>> params = {'C': [[1e3, 1e4, 1e5], [3, 11, 11], 'soft_float']}
            >>> agscv = AutoGSCV(estimator, params, total_passes=3,
            ...     total_passes_is_hard=True)
            >>> agscv.total_passes_is_hard
            True
            >>> agscv.total_passes_is_hard = False
            >>> agscv.total_passes_is_hard
            False

            When possible, if the parent GridSearch accepts a 'refit'
            kwarg and that value is not False, refit is deferred until
            the final pass to save time. For example, when the parent is
            a sci-kit GridSearch and refit is not False, AutoGridSearch
            disables refit until completion of the last pass. In this way,
            AutoGridSearch avoids unnecessary refits during intermediate
            passes and only performs the refit on the final best values.
            The dask GridSearch modules that accept a refit argument
            require that refit be set to True, so in that case every
            pass of AutoGridSearch must perform the refit.


            Terminology
            -----------
            Definitions for terms found in the autogridsearch docs.

            'Universal bound' - A logical bound for search spaces that is
            enforced thoughout the AutoGridSearch module. For soft and
            hard integers, the universal lower bound is 1; zero and
            negative numbers can never be included in a soft/hard integer
            search space. For soft and hard floats, the universal lower
            bound is zero; negative numbers can never be included in a
            soft/hard float search space. This bound has implications on
            how to perform a search over a boolean space; more on that
            elsewhere in the docs. AutoGridSearch will terminate if
            instructions are passed to the params argument that violate
            the universal bounds. There is no lower or upper universal
            bound for strings. There is no upper universal bound for
            integers or floats. There is no lower or upper universal
            bound for fixed search spaces. --- PIZZA verify this

            'fixed' parameter - A parameter whose search space is static.
            A 'string' parameter is a form of fixed parameter. The search
            space will not be shifted or drilled. The search grid
            provided at the start is the only search grid for every pass
            with one exception. The search space can only be changed by
            specifying a pass number on which to shrink the space to a
            single value thereafter (the best value from the preceding
            round.) Consider a search space over depth of a decision tree.
            A search space might be [3, 4, 5], where no other values are
            allowed to be searched. This would be a 'fixed_integer'
            search space.

            'hard' parameter - A parameter whose search is bounded to a
            contiguous subset of real numbers, observant of the universal
            hard bounds. The space will be drilled but cannot be shifted.
            The search space can be shrunk to a single value (i.e., the
            best value from the preceding round is the only value searched
            for all remaining rounds) by setting the 'points' for the
            appropriate round(s) to 1. Consider searching over l1_ratio
            for a sci-kit learn LogisticRegression classifier. Any real
            number in the interval [0, 1] is allowed. This is a
            'hard_float' search space.

            'soft' parameter - A parameter whose search space can be
            shifted and drilled, and is only bounded by the universal
            bounds. The search space can be shrunk to a single value
            (i.e., the best value from the preceding round is the only
            value searched for all remaining rounds) by setting the
            'points' for the appropriate round(s) to 1. Consider
            searching over regularization constant 'alphas' in a sci-kit
            learn RidgeCV estimator. Alpha can be any non-negative real
            number. A starting search space might be [1000, 2000, 3000].
            This a 'soft_float' search space.

            'shrink pass' -- pizza finish

            shift - The act of incrementing or decrementing all the values
            in a search grid by a fixed amount if GridSearchCV returns a
            best value that is on one of the edges of a given grid. This
            is best explained with an example. Consider a soft integer
            search space: grid = [20, 21, 22, 23, 24]. If the best value
            returned by GridsearchCV is 20, then a 'left-shift' is
            affected by decrementing every value in the grid by
            max(grid) - grid[1] -> 3. The search grid for the next round
            is [17, 18, 19, 20, 21]. Similarly, if the best value
            returned by GridsearchCV is 24, then a 'right-shift' is
            affected by incrementing every value in the grid by
            grid[-2] - min(grid) -> 3. The search grid for the next round
            is [23, 24, 25, 26, 27]. String, fixed, and hard spaces are
            not shifted. If passed any 'soft' spaces, autogridsearch
            will perform shifting passes until 1) it reaches a pass in
            which all soft parameters' best values simultaneously fall
            off the edges of their search grids, 2) 'max_shifts' is
            reached, or 3) total_passes_is_hard is True and total_passes
            is reached.

            drill - The act of narrowing a search space based on the best
            value returned from the last round of GridSearchCV and the
            grid used for that search. Not applicable to 'fixed' or
            'string' parameters. Briefly and simplistically, the next
            search grid is a 'zoom-in' on the last round's (sorted) grid
            in the region created by the search values that are
            adjacent to the best value. For float search spaces, all
            intervals are infinitely divisible and will be divided
            according to the number of points provided in 'params'. For
            integer search spaces, when the limit of unit intervals is
            approached, the search space is divided with unit intervals
            and the number of points to search is adjusted accordingly,
            regardless of the number of search points stated in params,
            and params is overwritten with the new number of points.

            linspace - a search space with intervals that are equal in
            linear space, e.g. [1,2,3]. See numpy.linspace.

            logspace - a search space whose log10 intervals are equal,
            e.g. [1, 10, 100]. See numpy.logspace.

            Restrictions
            ------------
            Integer search spaces must be greater than or equal to 1.
            Float search spaces must be greater than or equal to 0.
            'Soft' search grids must have at least 3 points.
            Logarithmic search intervals must be base 10 and searched
            values must be integers.

            Boolean Use Case
            ----------------
            To accomplish a search over boolean values, use 'fixed_float'
            and provide the search grid as [False, True] or [0, 1].
            This cannot be done with 'fixed_integer', as zero is not allowed
            in AutoGridSearch integer space.

            Params Argument
            ---------------
            'params' must be a single dictionary. AutoGridSearch cannot
            accomodate multiple params entries in the same way that
            sci-kit learn and dask GridSearchCV can accomodate multiple
            param_grids.

            The required argument 'params' must be of the following form:
            dict(
                'arg or kwarg name as string': list-type(...),
                'another arg or kwarg name as string': list-type(...),
                ...
            )

            The list-type field differs in construction for string and
            numerical parameters, and numerical parameters have two
            different variants that are both acceptable.

            ** * ** * **

            For string parameters, the list field is constructed as:
                [
                first search grid: list-like,
                shrink pass: int | None,
                'string': str
                ]
            E.g.:
                [['a', 'b', 'c'], 3, 'string']

            The list-like in the first position is the grid that will be
            used for all grid searches for this parameter, with the
            exception described below.

            The middle position ('shrink pass') must be an integer
            greater than one or None; if None, autogridsearch sets it
            to an arbitrary large integer. This value indicates the
            pass number on which to only select the single best value
            for that parameter out of best_params_ and proceed with grid
            searches using only that single value. Consider the following
            instructions [['a', 'b', 'c'], 4, 'string'], with
            total_passes = 5 and a true best value of 'c' that is
            correctly discovered by GridSearchCV.
            This will generate the following search grids:
            pass 1: ['a', 'b', 'c']; best value = 'c'
            pass 2: ['a', 'b', 'c']; best value = 'c'
            pass 3: ['a', 'b', 'c']; best value = 'c'
            pass 4: ['c']
            pass 5: ['c']
            This reduces the total searching time by minimizing the
            number of redundant searches.

            The text field 'string' is required for all string types,
            as it informs AutoGridSearch to handle the instructions
            and grids as string parameters.

            ** * ** * **

            ** * ** * **

            For numerical parameters, the list field can be constructed
            in two ways:

            1)
                [
                first search grid: list-like,
                number of points for each pass: int or list-like of ints,
                search type: str
                ]

                E.g. for 4 passes:
                    [[1, 2, 3], 3, 'fixed_integer']
                    or
                    [[1, 2, 3], [3, 3, 3, 3], 'fixed_integer']

                The list-like in the first position is the grid that will
                be used as the first search grid for this parameter.
                Because this is a fixed integer, this grid will be used
                for all searches, with one exception, explained below.

            2)
                [
                'logspace' or 'linspace': str,
                start_value: must be an integer if integer type or logspace,
                end_value: must be an integer if integer type or logspace,
                number of points for each pass: int or list-like of ints,
                search type: str
                ]

                E.g. for 4 passes:
                    ['linspace', 1, 5, 5, 'fixed_integer']
                    or
                    ['logspace', 0, 3, [4, 6, 6, 6], 'soft_float']

                    Effectively, this is the same as constructing the
                    param instructions in this way:
                    [numpy.linspace(1, 5, 5), 5, 'fixed_integer']
                    or
                    [numpy.logspace(0, 3, 4), [4, 6, 6, 6], 'soft_float']

                'logspace' or 'linspace' indicates the type of interval
                in the first grid
                start_value is the lowest value in the first grid
                end_value is the largest value in the first grid

            The second-to-last position of both constructs, 'number of
            points for each pass' must be an integer greater than zero
            or a list-type of such integers. If a single integer, this
            number will be the number of points in each grid for all
            searches after the first pass. If a list-type of integers,
            the length of the list-type must equal total_passes. The
            number of points for the first pass, although required to
            fulfill the length requirement, is effectively ignored and
            overwritten by the actual length of the first grid. Each
            subsequent value in the list-like dictates the number of
            points to put in the new grid for its respective pass. If any
            value in the list-like is entered as 1, all subsequent values
            must also be 1. For fixed spaces, the only acceptable entries
            are 1 or the length of the first (and only) grid. For integer
            spaces, the entered points are overwritten as necessary to
            maintain an integer space.

            If number of points is ever set to 1, the best value from the
            previous pass is used as the single search value in all
            subsequent search grids. This reduces the total searching
            time by minimizing the number of redundant searches.

            The last field for both constructs, 'search type', is required
            for all types, as it informs AutoGridSearch how to handle the
            instructions and grids. There are six allowed entries:
                'fixed_integer' - static grid of integers
                'fixed_float' - static grid of floats
                'hard_integer' - integer search space where the minimum
                    and maximum values of the first grid serve as bounds
                    for all searches
                'hard_float' - continuous search space where the minimum
                    and maximum values of the first grid serve as bounds
                    for all searches
                'soft_integer' - integer search space only bounded by the
                    universal minimum for integers
                'soft_float' - continous search space only bounded by the
                    universal minimum for floats

            ** * ** * **

            All together, a valid params argument for total_passes == 3
            might look like:
            {
                'solver': [['lbfgs', 'saga'], 2, 'string'],
                'max_depth': [[1, 2, 3, 4], [4, 4, 1], 'fixed_integer'],
                'C': [[1e1, 1e2, 1e3], [3, 11, 11], 'soft_float],
                'n_estimators': [[8, 16, 32, 64], [4, 8, 4], 'soft_integer'],
                'tol': ['logspace', 1e-6, 1e-1, [6, 6, 6], 'hard_float']
            }


            24_05_27 pizza does this say anything about logspace regap or
            logspace transition to linspace?


            Parameters
            ----------
            estimator:
                any estimator that follows the scikit-learn / dask
                fit / predict API. Includes scikit-learn, dask, lightGBM,
                and xgboost estimators.
            params:
                pizza put something here so hover-over has something
                see Params Argument
            total_passes:
                int, default 5 - the number of grid searches to perform.
                The actual number of passes run can be different from
                this number based on the setting for thetotal_passes_is_hard
                argument. If total_passes_is_hard is True, then the
                maximum number of total passes will always be the value
                assigned to total_passes. If total_passes_is_hard is
                False, a round that performs a 'shift' operation will
                increment the total number of passes, essentially causing
                shift passes to not count toward the total number of
                passes. Read elsewhere in the docs for more information
                about 'shifting' and 'drilling'.
            total_passes_is_hard:
                bool, default False - If True, total_passes is the exact
                number of grid searches that will be performed. If False,
                rounds in which a 'shift' takes place will increment
                the total passes, essentially causing 'shift' passes to
                be ignored against the total count of grid searches.
            max_shifts:
                [None, int], default None - The maximum number of
                'shifting' searches allowed. If None, there is no limit
                to the number of shifts that AutoGridSearch will perform.
            agscv_verbose:
                bool, default False - display status of AutoGridSearch
                and other helpful information during the grid searches,
                in addition to any verbosity displayed by the underlying
                GridsearchCV module.

            Attributes
            ----------
            estimator:
                estimator whose hyperparameters are to be optimized
            params:
                instructions for building param_grids
            total_passes:
                Minimum number of grid search passes to perform
            total_passes_is_hard:
                If True, total_passes is the actual number of grid
                searches performed. If False, total_passes is the minimum
                number of grid searches performed.
            max_shifts:
                The maximum allowed shifting passes to perform.
            agscv_verbose:
                =False,
            GRIDS_:
                Dictionary of param_grids run on each pass. As
                AutoGridSearch builds param_grids for each pass, they are
                stored in this attribute for later analysis. The keys of
                the dictionary are the zero-indexed pass number, i.e.,
                external pass number 2 is key 1 in this dictionary.
            RESULTS_:
                Dictionary of best_params_ for each pass. The keys of the
                dictionary are the zero-indexed pass number, i.e.,
                external pass number 2 is key 1 in this dictionary. The
                final key holds the most precise estimates of the best
                hyperparameters for the estimator.

            Examples
            --------
            >>> from pybear.model_selection import autogridsearch_wrapper
            >>> from sklearn.model_selection import GridSearchCV
            >>> from sklearn.linear_model import LogisticRegression
            >>> from sklearn.datasets import make_classification
            >>> AutoGridSearchCV = autogridsearch_wrapper(GridSearchCV)
            >>> estimator = LogisticRegression(
            ...     penalty="l2",
            ...     dual=False,
            ...     tol=1e-4,
            ...     C=1.0,
            ...     fit_intercept=True,
            ...     intercept_scaling=1,
            ...     class_weight=None,
            ...     random_state=None,
            ...     solver="lbfgs",
            ...     max_iter=100,
            ...     multi_class="auto",
            ...     verbose=0,
            ...     warm_start=False,
            ...     n_jobs=None,
            ...     l1_ratio=None,
            ... )
            >>> params = {
            ...     'C': [5_000, 10_000, 15_000],
            ...     'fit_intercept': [True, False],
            ...     'solver': ['lbfgs', 'saga'],
            ... }
            >>> gscv = AutoGridSearchCV(
            ...     estimator,
            ...     params,
            ...     total_passes=4,
            ...     total_passes_is_hard=True,
            ...     max_shifts=3,
            ...     agscv_verbose=False,
            ... )
            >>> X, y = make_classification(n_samples=10_000, n_features=100)
            >>> gscv.fit(X, y)


            """

            self.estimator = estimator
            self.params = params
            self.total_passes = total_passes
            self.total_passes_is_hard = total_passes_is_hard
            self.max_shifts = max_shifts
            self.agscv_verbose = agscv_verbose

            self._validation()

            _val_parent_gscv_kwargs(
                self.estimator, GridSearchParent, parent_gscv_kwargs
            )

            # super() instantiated in init() for access to GridSearchCV's
            # pre-run attrs and methods
            super().__init__(self.estimator, {}, **parent_gscv_kwargs)

            # THIS MUST STAY HERE FOR demo TO WORK
            self.reset()

        # END __init__() ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        # _validation() ################################################

        def _validation(self):

            """
            Validate the args and kwargs of the AutoGridSearch wrapper.
            Validation of the args and kwargs for the underlying GridSearch
            is handled by itself.

            """

            self.params, self.total_passes = \
                val_params_total_passes(self.params, self.total_passes)

            val_estimator(self.params, self.estimator)

            self.total_passes_is_hard = \
                val_total_passes_is_hard(self.total_passes_is_hard)

            self.max_shifts = val_max_shifts(self.max_shifts)

            self.agscv_verbose = val_agscv_verbose(self.agscv_verbose)

        # END _validation() ############################################


        # reset() ######################################################
        def reset(self) -> None:

            """
            Restore AutoGridSearch to pre-run state. Runs at the end of
            init and can be called as a method. Objects populated
            while AutoGridSearch runs are reset to pre-run condition.

            Return
            ------
            None


            """

            self._shift_ctr = 0

            # Create 2 dictionaries that hold information about starting /
            # calculated grids and best_params_
            # Assignments to self.GRIDS_ will be made in _get_next_param_grid()
            self.GRIDS_ = dict()
            self.RESULTS_ = dict()
            # best_params_ WILL BE ASSIGNED TO THE pass/idx IN
            # self.RESULTS_ WITHOUT MODIFICATION


            # IS_LOGSPACE IS DYNAMIC, WILL CHANGE WHEN A PARAM'S SEARCH
            # GRID INTERVAL IS UNITIZED OR TRANSITIONS FROM LOGSPACE TO
            # LINSPACE
            self._IS_LOGSPACE = _build_is_logspace(self.params)



            # ONLY POPULATE WITH numerical_params WITH "soft" BOUNDARY
            # AND START AS FALSE
            self._PHLITE = {}
            for hprmtr in self.params:
                if 'soft' in self.params[hprmtr][-1]:
                    self._PHLITE[hprmtr] = False

        # END reset() ##################################################


        def _get_next_param_grid(
            self,
            _pass: int,
            _best_params_from_previous_pass: BestParamsType
            ) -> None:

            """
            Core functional method. Bypassed on first pass (pass zero
            internally, pass 1 externally). Otherwise, populate GRIDS_
            with search grids for each parameter based on the previous
            round's best values returned from GridSearchCV and the
            instructions specified in :param: params.

            Parameters
            ----------
            _pass:
                int - internal iteration counter
            _best_params_from_previous_pass:
                dict[str, [int, float, str]] - best_params_ returned by
                Gridsearch for the previous pass

            Return
            ------
            None

            """

            self.GRIDS_, self.params, self._PHLITE, self._IS_LOGSPACE, \
                self._shift_ctr, self.total_passes = \
                    _get_next_param_grid(
                        self.GRIDS_,
                        self.params,
                        self._PHLITE,
                        self._IS_LOGSPACE,
                        _best_params_from_previous_pass,
                        _pass,
                        self.total_passes,
                        self.total_passes_is_hard,
                        self._shift_ctr,
                        self.max_shifts
                    )


        def demo(
                self,
                *,
                true_best_params: BestParamsType=None,
                mock_gscv_pause_time: Union[int, float]=5
            ):

            """
            Simulated trials of the AutoGridSearch instance. Visually
            inspect the generated grids and performance of the AutoGridSearch
            instance with its respective inputs in converging to the mock
            targets provided in true_best_params.

            Pizza add stuff, especially stuff about how to use.

            Parameters
            ----------
            true_best_params:
                dict[str, [int, float, str]] - dict of user-generated
                true best values for the parameters given in params.
                If not passed, random true best values are generated
                based on the first round grids made from the instructions
                in params.

            mock_gscv_pause_time:
                int, float - time in seconds to pause, simulating a trial
                of GridSearch

            Return
            ------
            -
                _DemoCls:
                    AutoGridSearchCV instance - The AutoGridSearch instance
                    used to run simulations, not the active instance of
                    AutoGridSearch.

            """

            _DemoCls = AutoGridSearch(
                self.estimator,  # must pass est to satisfy val even tho not used
                params=self.params,
                total_passes= self.total_passes,
                total_passes_is_hard=self.total_passes_is_hard,
                max_shifts=self.max_shifts,
                agscv_verbose=False
            )

            _demo(
                _DemoCls,
                _true_best=true_best_params,
                _mock_gscv_pause_time=mock_gscv_pause_time
            )

            return _DemoCls   # for test purposes only



        def print_results(self) -> None:

            """
            Print search grid and best value to the screen for all
            parameters in all passes.

            Return
            ------
            None

            """

            # CHECK IF fit() WAS RUN YET, IF NOT,
            # THROW GridSearch's "not fitted yet" ERROR
            self.best_score_
            _print_results(self.GRIDS_, self.RESULTS_)


        def fit(
            self,
            X: Iterable[Union[int, float]],
            y: Iterable[Union[int, float]]=None,
            groups=None,
            **params
            ):

            """
            Supercedes sklearn / dask GridSearchCV fit() method. Run
            underlying fit() method with all sets of parameters at least
            'total_passes' number of times.

            Parameters
            ----------
            X:
                array-like Iterable[int | float] - training data
            y:
                array-like Iterable[int | float], default None - target
                for training data
            groups:
                default None - Group labels for the samples used while
                splitting the dataset into train/test set

            Return
            ------
            -
                self: Instance of fitted estimator.

            See Also
            --------
            sklearn.model_selection.GridSearchCV.fit()
            dask_ml.model_selection.GridSearchCV.fit()


            """

            # this must be here because allowing attrs of AutoGridSearch
            # instance to be set directly
            self._validation()

            self.reset()

            _pass = 0
            while _pass < self.total_passes:

                if _pass == 0:
                    self.GRIDS_ = _build(self.params)
                else:
                    self._get_next_param_grid(
                        _pass,
                        _best_params_from_previous_pass
                    )

                if self.agscv_verbose:
                    print(f"\nPass {_pass+1} starting grids:")
                    for k, v in self.GRIDS_[_pass].items():
                        try: print(f'{k}'.ljust(15), np.round(v, 4))
                        except: print(f'{k}'.ljust(15), v)

                # Should GridSearchCV param_grid format ever change, code
                # would go here to adapt the _get_next_param_grid() output
                # to the required GridSearchCV.param_grid format
                _ADAPTED_GRIDS = deepcopy(self.GRIDS_)

                # (param_grid/param_distributions/parameters is over-
                # written and GSCV is fit()) total_passes times. After
                # a run of AutoGSCV, the final results held in the final
                # pass attrs and methods of GSCV are exposed by the
                # AutoGridSearch instance.
                # pizza these trys can probably come out
                try:
                    self.param_grid = _ADAPTED_GRIDS[_pass]
                except:
                    pass
                try:
                    self.param_distributions = _ADAPTED_GRIDS[_pass]
                except:
                    pass
                try:
                    self.parameters = _ADAPTED_GRIDS[_pass]
                except:
                    pass

                del _ADAPTED_GRIDS

                # 24_02_28_07_25_00 if sklearn parent GridSearchCV has refit
                # and refit is not False, only refit on the last pass --
                # 24_06_01_11_48_00 dask GridSearchCV and RandomizedSearchCV
                # require refit always be True, IncrementalSearchCV,
                # HyperbandSearchCV, SuccessiveHalvingSearchCV and
                # InverseDecaySearchCV do not take a refit kwarg

                _is_dask = 'DASK' in str(type(self.estimator)).upper()
                if not _is_dask and hasattr(self, 'refit'):
                    if _pass == 0:
                        original_refit = self.refit
                        self.refit = False
                    elif _pass == self.total_passes - 1:
                        self.refit = original_refit
                        del original_refit
                del _is_dask

                if self.agscv_verbose:
                    print(f'Running GridSearch... ', end='')

                super().fit(X, y=y, groups=groups, **params)

                if self.agscv_verbose:
                    print(f'Done.')

                _best_params = self.best_params_

                if self.agscv_verbose:
                    print(f"Pass {_pass + 1} best params: ", end='')
                    _output = []
                    for k,v in _best_params.items():
                        try: _output.append(f"{k}: {v:,.4f}")
                        except: _output.append(f"{k}: {v}")
                    print(", ".join(_output))
                    del _output

                # Should GridSearchCV best_params_ format ever change,
                # code would go here to adapt the GridSearchCV.best_params_
                # output to the required self.RESULTS_ format
                adapted_best_params = deepcopy(_best_params)

                self.RESULTS_[_pass] = adapted_best_params
                _best_params_from_previous_pass = adapted_best_params
                # _bpfpp GOES BACK INTO self.get_next_param_grid

                del adapted_best_params

                _pass += 1

            del _best_params_from_previous_pass

            if self.agscv_verbose:
                print(f"\nfit successfully completed {self.total_passes} "
                      f"pass(es) with {self._shift_ctr} shift pass(es).")


            return self

    # pizza this isnt working
    AutoGridSearch.__doc__ = AutoGridSearch.__init__.__doc__

    return AutoGridSearch






















































