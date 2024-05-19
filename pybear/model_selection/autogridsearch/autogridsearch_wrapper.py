import time
import numpy as np
from dask import compute


from ._autogridsearch_wrapper._validation._agscv_verbose import _agscv_verbose as val_agscv_verbose
from ._autogridsearch_wrapper._validation._estimator import _estimator as val_estimator
from ._autogridsearch_wrapper._validation._max_shifts import _max_shifts as val_max_shifts
from ._autogridsearch_wrapper._validation._params import _params_validation as val_params
from ._autogridsearch_wrapper._validation._total_passes import _total_passes as val_total_passes
from ._autogridsearch_wrapper._validation._total_passes_is_hard import _total_passes_is_hard as val_total_passes_is_hard
from ._autogridsearch_wrapper._build_first_grid_from_params import _build
from ._autogridsearch_wrapper._build_is_logspace import _build_is_logspace





# POTENTIAL PARENTS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
#                                       # ARGS                          # KWARGS
# sklearn.model_selection
# --- GridSearchCV                estimator, param_grid              scoring, n_jobs, cv, refit, verbose, +others
# --- RandomizedSearchCV          estimator, param_distributions     scoring, n_jobs, cv, refit, verbose, +others
# --- HalvingGridSearchCV         estimator, param_grid              scoring, n_jobs, cv, refit, verbose, +others
# --- HalvingRandomSearchCV       estimator, param_distributions     scoring, n_jobs, cv, refit, verbose, +others

# dask_ml.model_selection
# --- GridSearchCV                estimator, param_grid              scoring, n_jobs, cv, refit, +others
# --- RandomizedSearchCV          estimator, param_distributions     scoring, n_jobs, cv, refit, +others
# --- IncrementalSearchCV         estimator, parameters              scoring, +others
# --- HyperbandSearchCV           estimator, parameters              scoring, +others
# --- SuccessiveHalvingSearchCV   estimator, parameters              scoring, +others
# --- InverseDecaySearchCV        estimator, parameters              scoring, +others

# other
# --- GridSearchCVThreshold       estimator, param_grid              scoring, n_jobs, cv, refit, verbose, +others
# END POTENTIAL PARENTS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



def autogridsearch_wrapper(GridSearchParent):

    class AutoGridSearch(GridSearchParent):

        """
        On first pass, get_next_param_grid returns the grids as instructed
        in the 'numerical_params' and 'string_params' parameters. On
        subsequent passes, get_next_param_grid returns new calculated
        grids based on results within 'best_params_'

        dask/sklearn.GridSearchCV.best_params_ RETURNS A DICT OF params
        AND THEIR RESPECTIVE OPTIMAL SETTINGS. THE WORKING METHOD OF THIS
        CLASS MUST TAKE IN THE best_params_ DICT AND RETURN ANOTHER DICT
        WITH RANGES OF NEW GRID VALUES THAT LOOKS LIKE param_grid from
        dask/sklearn GridSearchCV, e.g.:
            {
            'C': [0,5,10],
            'l1_ratio': [0, 0.5, 1],
            'solver': ['lbfgs', 'saga']
            }

        numerical_params (IF PASSED) MUST BE A DICT THAT LOOKS LIKE
        {
            'param1': [
                        'logspace/linspace',
                        start_value,  (must be an integer if logspace)
                        end_value,  (must be an integer if logspace)
                        number_of_points as (i) int >= 1 or (ii) list of
                        ints >=1 with len==total_passes,
                        'soft/hard/fixed_int/float' (start/end values can
                         slide or are static for float/int, or grid does
                         not change)
                       ]
            - or -
            'param1': [
                        first_grid as list of grid values for first search,
                        number_of_points as (i) int >= 1 or (ii) list of
                        ints >=1 with len==total_passes,
                        'soft/hard/fixed_int/float' (start/end values
                        can slide or are static for float/int, or grid
                        does not change)
                        ]
        }

        string_params (IF PASSED) MUST BE A DICT THAT LOOKS LIKE
        {
            'param1': [*list of estimator (kw)args]
            or
            'param1': [[*list of estimator (kw)args]]
            or
            'param1': [
                       [*list of estimator (kw)args],
                       pass_after_which_to_only_run_the_single_best_value
                       ],
        }
        """

        def __init__(
                        self,
                        estimator,
                        # PIZZA GridSearchCV CAN TAKE param_grid as {} or [{},{},..]... WHAT TO DO :(
                        numerical_params:dict=None,
                        string_params:dict=None,

                        # PIZZA THINK ABOUT IF HARD OR FIXED INTEGERS CAN EVER BE NEGATIVE (THEY BYPASS SHIFT AND GO STR8 TO DIGGER)
                        # AKA CAN DIGGER HANDLE NEGATIVE INTEGERS
                        # PIZZA THINK ABOUT IF HARD OR FIXED FLOATS CAN EVER BE NEGATIVE (THEY BYPASS SHIFT AND GO STR8 TO DIGGER)
                        # AKA CAN DIGGER HANDLE NEGATIVE FLOATS

                        total_passes:int=5,
                        total_passes_is_hard:bool=False,
                        max_shifts:int=10,
                        agscv_verbose:bool=0,

                        **parent_gscv_kwargs
            ):

            # init VALIDATION #################################################

            _estimator = val_estimator(_estimator)

            _params = val_params(_params)

            _total_passes = val_total_passes(_total_passes)

            _total_passes_is_hard = val_total_passes_is_hard(
                _total_passes_is_hard)

            _max_shifts = val_max_shifts(_max_shifts)

            _agscv_verbose = val_agscv_verbose(_agscv_verbose)

            # END init VALIDATION #############################################

            # super() instantiated in init() for access to XGridSearchCV's pre-run attrs and methods
            super().__init__(estimator, {}, **parent_gscv_kwargs)

            # THIS MUST STAY HERE FOR demo TO WORK
            self.reset()

        # END __init__() ######################################################
        #######################################################################
        #######################################################################


        # reset() #############################################################
        def reset(self):

            """
            :return: <nothing>

            Restore AutoGridSearch to pre-run state. Runs at the end of init and can be called as a method. Objects populated
            while AutoGridSearch runs are reset to pre-run condition.
            """

            self.shift_ctr = 0

            # CREATE TWO DICTIONARIES THAT HOLD INFORMATION ABOUT STARTING/CALCULATED GRIDS AND best_params_
            # ASSIGNMENTS TO self.GRIDS WILL BE MADE IN get_next_param_grid()
            self.GRIDS = dict()
            self.RESULTS = dict()
            # best_params_ WILL BE ASSIGNED TO THE pass/idx IN self.RESULTS WITHOUT MODIFICATION


            # IS_LOGSPACE IS DYNAMIC, WILL CHANGE WHEN A PARAM'S SEARCH GRID INTERVAL IS
            # UNITIZED OR TRANSITIONS FROM LOGSPACE TO LINSPACE
            self.IS_LOGSPACE = _build_is_logspace(self._params)



            # ONLY POPULATE WITH numerical_params WITH "soft" BOUNDARY AND START AS FALSE
            self.PARAM_HAS_LANDED_INSIDE_THE_EDGES = \
                {hprmtr: False for hprmtr in self._params if 'soft' in self._params[hprmtr][-1]}

        # END reset() #########################################################


        #######################################################################
        # FUNCTIONAL METHOD ###################################################
        def get_next_param_grid(self, _pass, best_params_from_previous_pass):


            """
            Core functional method. On first pass (pass zero), populate
            GRIDS with the search grids as specified in :param: params.
            For subsequent passes, generate new grids based on the
            previous grid (as held in GRIDS) and its associated
            best_params_ returned from GridSearchCV.

            Parameters
            ----------
            _pass:
                int - iteration counter
            :param
                _best_params_from_previous_pass: dict - best_params_
                returned by Gridsearch for the previous pass

            Return
            ------
            pizza_grid:
                dict of grids to be passed to GridSearchCV for the next
                pass, must be in param_grid format

            """

            # PIZZA --- RETURN FROM FUNCTION _get_next_param_grid
            _GRIDS, _params = _get_next_param_grid(

            )


        # END FUNCTIONAL METHOD ########################################################################################################
        ################################################################################################################################


        def demo(self, true_best_params=None):

            """
            :param true_best_params: dict - dict of user-generated true best parameters for the given numerical / string params.
                                     If not passed, _random_ best params are generated.
            :return: <nothing>

            Visually inspect the generated grids and performance of AutoGridSearch with the user-given numerical_params,
            string_params, and total_passes in converging to the targets in true_best_params.
            """

            demo_cls = AutoGridSearch(
                                        'dummy_estimator',
                                        numerical_params=self.numerical_params,
                                        string_params=self.string_params,
                                        total_passes= self.total_passes,
                                        total_passes_is_hard=self.total_passes_is_hard,
                                        max_shifts=self.max_shifts,
            )

            # ##############################################################################################################
            # STUFF FOR MIMICKING GridSearchCV OUTPUT ######################################################################
            if true_best_params is None:
                true_best_params = dict()
                for _param in demo_cls.numerical_params | demo_cls.string_params:
                    if _param in demo_cls.numerical_params:
                        __ = demo_cls.numerical_params[_param]
                        _min, _max = min(__[0]), max(__[0])
                        _gap = _max - _min
                        if __[-1]=='hard_float':
                            true_best_params[_param] = np.random.uniform(_min, _max, size=(1,))[0]
                        elif __[-1]=='hard_integer':
                            true_best_params[_param] = np.random.choice(
                                                                        np.arange(_min, _max+1, 1, dtype=np.int32),
                                                                        1,
                                                                        replace=False
                            )[0]
                        elif __[-1]=='fixed_float':
                            true_best_params[_param] = np.random.choice(__[0], 1, replace=False)[0]
                        elif __[-1]=='fixed_integer':
                            true_best_params[_param] = np.random.choice(__[0], 1, replace=False)[0]
                        elif __[-1]=='soft_float':
                            true_best_params[_param] = np.random.uniform(max(_min-_gap, 0), _max+_gap, size=(1,))[0]
                        elif __[-1].lower()=='soft_integer':
                            true_best_params[_param] = \
                                np.random.choice(
                                                 np.linspace(max(_min-_gap, 1),
                                                             _max+_gap,
                                                             int(_max+_gap-max(_min-_gap, 1)+1)
                                                             ).astype(int),
                                                 1,
                                                 replace=False
                            )[0]

                        del __, _min, _max, _gap

                    elif _param in demo_cls.string_params:
                        true_best_params[_param] = np.random.choice(demo_cls.string_params[_param][0], 1, replace=False)[0]


            # VALIDATE true_best_params ################################################################################
            str_exc = Exception(f"{_param}: true_best_params string params must be a single string that is in allowed values")
            for _param in demo_cls.numerical_params | demo_cls.string_params:
                _best = true_best_params[_param]
                if _param not in true_best_params:
                    raise Exception(f"{_param}: true_best_params must contain all the params that are in numerical & string params")

                if _param in demo_cls.numerical_params:
                    _npr = demo_cls.numerical_params[_param]
                    if True not in [x in str(type(_best)).lower() for x in ['int','float']]:
                        raise Exception(f"{_param}: true_best_params num params must be a single number")
                    if demo_cls.IS_LOGSPACE[_param] and _best < 0:
                        raise Exception(f"{_param}: true_best_param must be > 0 for logspace search")
                    if True in [x in _npr[-1] for x in ['hard', 'fixed']]:
                        if not _best >= min(_npr[0]) and not _best <= max(_npr[0]):
                            raise Exception(f"{_param}: 'hard' and 'fixed' numerical true_best_param must be in range of given allowed values")
                    else:  # IS SOFT NUMERIC
                        if 'integer' in _npr[-1] and _best < 1: raise Exception(f"{_param}: soft integer best value must be >= 1")
                        elif 'float' in _npr[-1] and _best < 0: raise Exception(f"{_param}: soft float best value must be >= 0")

                    del _npr
                elif _param in demo_cls.string_params:
                    if not 'str' in str(type(_best)).lower(): raise str_exc
                    if not _best in demo_cls.string_params[_param][0]: raise str_exc

            del str_exc, _best
            # END VALIDATE true_best_params ############################################################################

            # FXN TO DISPLAY THE GENERATED true_best_params ####################################################################
            print(f'\nGenerated true best params are:')
            def display_true_best_params(true_best_params):
                NUM_TYPES, STRING_TYPES = [], []
                for _ in true_best_params:
                    if _ in demo_cls.numerical_params: NUM_TYPES.append(_)
                    elif _ in demo_cls.string_params: STRING_TYPES.append(_)
                print(f'\nNumerical hyperparameters:')
                for _ in NUM_TYPES:
                    print(f'{_}:'.ljust(20) + f'{true_best_params[_]}')
                print(f'\nString hyperparameters:')
                for _ in STRING_TYPES:
                    print(f'{_}:'.ljust(20) + f'{true_best_params[_]}')
                print()
                del NUM_TYPES, STRING_TYPES

            display_true_best_params(true_best_params)
            # END FXN TO DISPLAY THE GENERATED true_best_params ###################################################################
            # END STUFF FOR MIMICKING GridSearchCV OUTPUT ##################################################################
            # ##############################################################################################################

            # MIMIC fit() FLOW AND OUTPUT
            # fit():
            #             1) run passes of GridSearchCV
            #               - 1a) get_next_param_grid()
            #               - 1b) fit GridSearchCV with next_param_grid
            #               - 1c) update self.RESULTS
            #             2) return best_estimator_

            # 1) run passes of GridSearchCV
            best_params = None
            _pass = 0
            while _pass < demo_cls.total_passes:
                print(f"\nStart pass {_pass+1} ###############################################################################")
                # 1a) get_next_param_grid()
                print(f'Building param grid... ', end='')
                demo_cls.get_next_param_grid(_pass, best_params_from_previous_pass=best_params)
                print(f'Done.')

                # print(f"\nPass {_pass+1}: grid coming out of get_next_param_grid going into fake GridSearchCV")

                def padder(words):
                    try: return str(words)[:13].ljust(15)
                    except: return 'NA'

                print(padder('param'), padder('type'), padder('true_best'), padder('previous_best'), padder('new_points'), padder('next_grid'))
                for _ in demo_cls.GRIDS[_pass]:
                    if _pass != 0: _best_params_ = demo_cls.RESULTS[_pass-1][_]
                    else: _best_params_ = 'NA'
                    if _ in self.numerical_params: _type_ = demo_cls.numerical_params[_][-1]
                    else: _type_ = 'string'
                    print(
                            padder(_),
                            padder(_type_),
                            padder(true_best_params[_]),
                            padder(_best_params_),
                            padder(len(demo_cls.GRIDS[_pass][_])),
                            f'{demo_cls.GRIDS[_pass][_]}'
                    )
                del padder, _best_params_, _type_

                # 1b) fit GridSearchCV with next_param_grid
                # SIMULATE WORK BY dask/sklearnGridSearchCV ON AN ESTIMATOR ##################################################
                combinations = np.prod(list(map(len, demo_cls.GRIDS[_pass].values())))
                print(f'\nThere are {combinations:,.0f} combinations to run')
                print(f"Simulating dask/sklearn GridSearchCV running on pass {_pass+1}...")
                time.sleep(5)  #(combinations)
                del combinations

                # CALCULATE WHAT THE best_params_ SHOULD BE BASED ON THE true_best_params. USE MIN LSQ FOR NUMERICAL,
                # FOR A STR PARAM, MAKE IT 10% CHANCE THAT THE RETURNED "best" IS A NON-BEST OPTION
                best_params = dict()
                for _param in demo_cls.GRIDS[_pass]:
                    __ = np.array(demo_cls.GRIDS[_pass][_param])
                    if _param in demo_cls.numerical_params:
                        if len(__)==1: best_params[_param] = __[0]
                        else:
                            LSQ = np.power(__ - true_best_params[_param], 2, dtype=np.float64)
                            best_params[_param] = __[LSQ==np.min(LSQ)][0]
                            del LSQ
                    elif _param in demo_cls.string_params:
                        if len(__)==1: best_params[_param] = __[0]
                        else:
                            best_params[_param] = np.random.choice(
                                    __,
                                    1,
                                    False,
                                    p=[0.9 if i == true_best_params[_param] else (1 - 0.9) / (len(__) - 1) for i in __]
                            )[0]

                    del __
                # END SIMULATE WORK BY dask/sklearnGridSearchCV ON AN ESTIMATOR ##############################################

                # 1c) update self.RESULTS
                demo_cls.RESULTS[_pass] = best_params
                print(f"\nEnd pass {_pass + 1} ###############################################################################")

                _pass += 1


            print(f'\nRESULTS:')
            for _pass in demo_cls.RESULTS:
                print(f'\nPass {_pass + 1} results:')
                for _param in demo_cls.RESULTS[_pass]:
                    print(
                            f' ' * 5 + f'{_param}:'.ljust(15) +
                            f'{demo_cls.GRIDS[_pass][_param]}'.ljust(90) +
                            f'Result = {demo_cls.RESULTS[_pass][_param]}'
                    )


            # DISPLAY THE GENERATED true_best_params AGAIN ####################################################################
            display_true_best_params(true_best_params)
            # END DISPLAY THE GENERATED true_best_params AGAIN #################################################################

            # 2) return best_estimator_ --- DONT HAVE AN ESTIMATOR TO RETURN

            print(f"demo fit successfully completed {demo_cls.total_passes} pass(es) with {demo_cls.shift_ctr} shift pass(es).")

            del demo_cls, best_params, true_best_params, display_true_best_params


        def print_results(self):

            """
            Print all results to the screen. Called by demo() and can be run as a method.
            :return: <nothing>
            """

            # CHECK IF fit() WAS RUN YET, IF NOT, THROW GridSearch's "not fitted yet" ERROR
            self.best_score_

            for _pass in self.RESULTS:
                print(f'\nPass {_pass + 1} results:')
                for _param in self.RESULTS[_pass]:
                    print(
                            f' ' * 5 + f'{_param}:'.ljust(15) +
                            f'{self.GRIDS[_pass][_param]}'.ljust(90) +
                            f'Result = {self.RESULTS[_pass][_param]}'
                    )


        def fit(self, X, y=None, groups=None, **params):
            """
            :param X: training data
            :param y: target for training data
            :param groups: Group labels for the samples used while splitting the dataset into train/test set
            :return: Instance of fitted estimator.

            Analog to dask/sklearn GridSearchCV fit() method. Run fit with all sets of parameters.
            """

            self.reset()

            _pass = 0
            while _pass < self.total_passes:

                if _pass == 0:
                    _param_grid = _build(_params)
                else:
                    _param_grid = self.get_next_param_grid(_GRIDS, _pass, _best_params_from_previous_pass)

                if self.agscv_verbose:
                    print(f"\nPass {_pass+1} starting grids:")
                    for k, v in _param_grid[_pass].items():
                        try: print(f'{k}'.ljust(15), np.round(v, 4))
                        except: print(f'{k}'.ljust(15), v)

                # SHOULD GridSearchCV param_grid FORMAT EVER CHANGE, CODE WOULD GO HERE TO ADAPT THE get_next_param_grid
                # OUTPUT TO THE REQUIRED XGridSearchCV.param_grid FORMAT
                adapted_param_grid = _param_grid

                # (param_grid/param_distributions/parameters is overwritten and X_GSCV is fit()) total_passes times.
                # After a run of AutoGSCV, the final results held in the final pass attrs and methods of X_GSCV are
                # exposed by the AutoGridSearch instance.
                try: self.param_grid = adapted_param_grid[_pass]
                except: pass
                try: self.param_distributions = adapted_param_grid[_pass]
                except: pass
                try: self.parameters = adapted_param_grid[_pass]
                except: pass

                del adapted_param_grid

                # 24_02_28_07_25_00 IF PARENT GridSearchCV HAS REFIT AND REFIT IS NOT False, ONLY REFIT ON THE LAST PASS
                if hasattr(self, 'refit'):
                    if _pass == 0:
                        original_refit = self.refit
                        self.refit = False
                    elif _pass == self.total_passes - 1:
                        self.refit = original_refit
                        del original_refit


                if self.agscv_verbose: print(f'Running GridSearch... ', end='')

                super().fit(X, y=y, groups=groups, **params)

                if self.agscv_verbose: print(f'Done.')

                _best_params = self.best_params_

                if self.agscv_verbose:
                    print(f"Pass {_pass + 1} best params: ", end='')
                    _output = []
                    for k,v in _best_params.items():
                        try: _output.append(f"{k}: {v:,.4f}")
                        except: _output.append(f"{k}: {v}")
                    print(", ".join(_output))
                    del _output

                # SHOULD GridSearchCV best_params_ FORMAT EVER CHANGE, CODE WOULD GO HERE TO ADAPT THE GridSearchCV.best_params_
                # OUTPUT TO THE REQUIRED self.RESULTS FORMAT
                adapted_best_params = _best_params

                self.RESULTS[_pass] = adapted_best_params
                _best_params_from_previous_pass = adapted_best_params  # GOES BACK INTO self.get_next_param_grid
                del adapted_best_params

                _pass += 1

            del _best_params_from_previous_pass

            if self.agscv_verbose:
                print(f"\nfit successfully completed {self.total_passes} pass(es) with {self.shift_ctr} shift pass(es).")


            return self


    return AutoGridSearch























































