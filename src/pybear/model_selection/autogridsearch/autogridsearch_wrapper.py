# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Sequence
from typing_extensions import Union, TypeAlias

import numbers
from copy import deepcopy
import numpy as np

from . import autogridsearch_docs

from ._autogridsearch_wrapper._type_aliases import BestParamsType

from ._autogridsearch_wrapper._print_results import _print_results

from ._autogridsearch_wrapper._validation._agscv_verbose import \
    _agscv_verbose as val_agscv_verbose
from ._autogridsearch_wrapper._validation._estimator import _estimator \
    as val_estimator
from ._autogridsearch_wrapper._validation._is_dask_gscv import _is_dask_gscv \
    as val_dask_gscv
from ._autogridsearch_wrapper._validation._max_shifts import _max_shifts \
    as val_max_shifts
from ._autogridsearch_wrapper._validation._parent_gscv_kwargs import \
    _val_parent_gscv_kwargs
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

from ..GSTCV._GSTCV import GSTCV
from ..GSTCV._GSTCVDask import GSTCVDask



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
    type(GSTCV),
    type(GSTCVDask)
]





def autogridsearch_wrapper(
    GridSearchParent: GridSearchType
) -> GridSearchType:

    """
    Wrap a sci-kit learn or dask GridSearchCV class with a class that
    overwrites the fit method of GridSearchCV. The superseding fit
    method automates multiple calls to the super fit() method using
    progressively more precise search grids based on previous search
    results. All sci-kit, dask, and pybear GridSearch modules are
    supported. See the sci-kit, dask, and pybear documentation for more
    information about the available GridSearchCV modules.


    Parameters
    ----------
    GridSearchParent:
        Sci-kit or dask GridSearchCV class, not instance.


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
    pybear.model_selection.GSTCV
    pybear.model_selection.GSTCVDask
    pybear.model_selection.AutoGSTCV
    pybear.model_selection.AutoGSTCVDask

    """


    class AutoGridSearch(GridSearchParent):

        def __init__(
            self,
            estimator,
            params: dict[
                str,
                list[Sequence[any], Union[int, Sequence[int]], str]
            ],
            *,
            total_passes: int=5,
            total_passes_is_hard: bool=False,
            max_shifts: Union[None, numbers.Integral]=None,
            agscv_verbose: bool=False,
            **parent_gscv_kwargs
            ) -> None:

            self.estimator = estimator
            self.params = params
            self.total_passes = total_passes
            self.total_passes_is_hard = total_passes_is_hard
            self.max_shifts = max_shifts
            self.agscv_verbose = agscv_verbose

            self._validation()

            parent_gscv_kwargs = _val_parent_gscv_kwargs(
                GridSearchParent, parent_gscv_kwargs
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
            -
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
                dict[str, Union[numbers.Real, str]] - best_params_
                returned by Gridsearch for the previous pass

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
            true_best_params: dict[str, any]=None,
            mock_gscv_pause_time: numbers.Real=5
        ):

            """
            Simulated trials of this AutoGridSearch instance.

            Demonstrate and assess AutoGridSearch's ability to generate
            appropriate grids with the given parameters in this
            AutoGridSearch instance (params, etc.) against mocked true
            best values. Visually inspect the generated grids and
            performance of the AutoGridSearch instance in converging to
            the mock targets provided in true_best_params. If no true
            best values are provided to true_best_params, random true
            best values are generated from the set of first search grids
            provided in params.

            Parameters
            ----------

            true_best_params:
                dict[str, Union[numbers.Real, str]] - dict of mocked
                true best values for an estimator's hyperparameters, as
                provided by the user or generated randomly. If not passed,
                random true best values are generated based on the first
                round grids made from the instructions in params.

            mock_gscv_pause_time:
                numbers.Real - time in seconds to pause, simulating a
                trial of GridSearch

            Return
            ------
            -
                _DemoCls:
                    AutoGridSearchCV instance - The AutoGridSearch instance
                    created to run simulations, not the active instance of
                    AutoGridSearch. This return is integral for tests of
                    the demo functionality, but has no other internal use.

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

            return _DemoCls   # for tests purposes only



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
            **fit_params
        ):

            """
            Supersedes sklearn / dask GridSearchCV fit method. Run
            underlying fit method with all sets of parameters at
            least :param: `total_passes` number of times.


            Parameters
            ----------
            X:
                array-like - training data
            y:
                array-like, default=None - target for training data
            groups:
                default=None - Group labels for the samples used while
                splitting the dataset into train/tests set


            Return
            ------
            -
                self: AutoGridSearch instance.


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
                        try:
                            print(f'{k}'.ljust(15), np.round(v, 4))
                        except:
                            print(f'{k}'.ljust(15), v)

                # Should GridSearchCV param_grid format ever change, code
                # would go here to adapt the _get_next_param_grid() output
                # to the required GridSearchCV.param_grid format
                _ADAPTED_GRIDS = deepcopy(self.GRIDS_)

                # (param_grid/param_distributions/parameters is over-
                # written and GSCV is fit()) total_passes times. After
                # a run of AutoGSCV, the final results held in the final
                # pass attrs and methods of GSCV are exposed by the
                # AutoGridSearch instance.

                self.param_grid = _ADAPTED_GRIDS[_pass]
                self.param_distributions = _ADAPTED_GRIDS[_pass]
                self.parameters = _ADAPTED_GRIDS[_pass]

                del _ADAPTED_GRIDS

                # *** ONLY REFIT ON THE LAST PASS TO SAVE TIME WHEN POSSIBLE ***
                # IS POSSIBLE WHEN:
                # sklearn or pybear parent -
                # == has refit and refit is not False
                # == is using only one scorer
                # IS NOT POSSIBLE WHEN:
                # == total_passes = 1
                # == using dask GridSearchCV or RandomizedSearchCV, they
                # require refit always be True to expose best_params_.
                # == dask IncrementalSearchCV, HyperbandSearchCV,
                # SuccessiveHalvingSearchCV and InverseDecaySearchCV do
                # not take a refit kwarg.
                # == When using multiple scorers, refit must always be
                # left on because multiple scorers dont expose best_params_
                # when multiscorer and refit=False
                try:
                    _multimetric = not (
                        callable(self.scoring) \
                        or isinstance(self.scoring, (str, type(None))) \
                        or len(self.refit) == 1
                    )
                except:
                    _multimetric = True  # maybe not really True, but
                    # since cant determine it, make agscw do refit
                    # everytime, if refit not False

                if hasattr(self, 'refit') \
                    and not self.total_passes == 1 \
                    and not val_dask_gscv(GridSearchParent) \
                    and not _multimetric:
                    if _pass == 0:
                        original_refit = self.refit
                        self.refit = False
                    elif _pass == self.total_passes - 1:
                        self.refit = original_refit
                        del original_refit
                # *** END ONLY REFIT ON THE LAST PASS TO SAVE TIME ********

                if self.agscv_verbose:
                    print(f'Running GridSearch... ', end='')

                try:
                    super().fit(X, y=y, groups=groups, **fit_params)
                except:
                    super().fit(X, y=y, **fit_params)

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


    # pizza this is likely being correctly assigned but pycharm tool tip
    # does not show it
    AutoGridSearch.__doc__ = autogridsearch_docs.__doc__
    AutoGridSearch.__init__.__doc__ = autogridsearch_docs.__doc__


    return AutoGridSearch






















































