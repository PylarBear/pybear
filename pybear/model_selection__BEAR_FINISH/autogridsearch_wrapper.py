import time
from dask import compute


# AutoGridSearch ran successfully in a real test outside of demo() on 24_02_10_14_07_00 !!!

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
        On first pass, get_next_param_grid returns the grids as instructed in the 'numerical_params' and 'string_params' parameters.
        On subsequent passes, get_next_param_grid returns new calculated grids based on results within 'best_params_'

        dask/sklearn.GridSearchCV.best_params_ RETURNS A DICT OF params AND THEIR RESPECTIVE OPTIMAL SETTINGS.
        THE WORKING METHOD OF THIS CLASS MUST TAKE IN THE best_params_ DICT AND RETURN ANOTHER DICT
        WITH RANGES OF NEW GRID VALUES THAT LOOKS LIKE param_grid from dask/sklearn GridSearchCV, e.g.:
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
                        number_of_points as (i) int >= 1 or (ii) list of ints >=1 with len==total_passes,
                        'soft/hard/fixed_int/float' (start/end values can slide or are static for float/int,
                                                            or grid does not change)
                       ]
            - or -
            'param1': [
                        first_grid as list of grid values for first search,
                        number_of_points as (i) int >= 1 or (ii) list of ints >=1 with len==total_passes,
                        'soft/hard/fixed_int/float' (start/end values can slide or are static for float/int,
                                                            or grid does not change)
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
                        # BEAR GridSearchCV CAN TAKE param_grid as {} or [{},{},..]... WHAT TO DO :(
                        numerical_params:dict=None,
                        string_params:dict=None,

                        # BEAR THINK ABOUT GETTING RID OF numerical_params AND string_params AND JUST REQUIRE
                        # ENTRY OF 'string' IN (NEW) LAST POSITION OF CURRENT string_params
                        # THIS WOULD ENABLE PASS OF SINGLE DICTS, AND LISTS OF SINGLE DICTS!!!!

                        total_passes:int=5,
                        total_passes_is_hard:bool=False,
                        max_shifts:int=10,
                        agscv_verbose:[int,bool]=0,

                        **parent_gscv_kwargs
            ):

            # init VALIDATION ###########################################################################################################

            # ABBREVIATED EXCEPTION HANDLING
            def _exc(text):
                raise Exception(f'{text}')

            self.agscv_verbose = agscv_verbose

            # VALIDATE total_passes
            if total_passes not in range(1, 100): raise _exc(f'total_passes must be an integer and 1 <= total_passes < 100')
            self.total_passes = total_passes

            # VALIDATE total_passes_is_hard
            if not isinstance(total_passes_is_hard, bool): raise _exc(f'total_passes_is_hard must be bool')
            self.total_passes_is_hard = total_passes_is_hard

            # VALIDATE max_shifts
            if max_shifts not in range(0, 100): raise _exc(f'max_shifts must be an integer and 0 <= max_shifts < 100')
            self.max_shifts = max_shifts

            _type = lambda x: str(type(x)).lower()
            _msg = f"numerical_params and string_params must be dictionaries or not passed"
            if True not in [y in _type(numerical_params) for y in ['dict', 'none']]: _exc(_msg)
            if True not in [y in _type(string_params) for y in ['dict', 'none']]: _exc(_msg)
            del _type, _msg

            # AT LEAST ONE OF numerical_params AND string_params MUST BE PASSED
            if numerical_params is None:
                if string_params is None: _exc(f'at least one of, or both of, numerical_params and string_params must be passed')
                numerical_params = {}
            elif string_params is None: string_params = {}

            # MUST BE AT LEAST 1 PARAM
            if len(numerical_params) + len(string_params) == 0:
                _exc(f"must be at least 1 param")

            # CHECK FOR DUPLICATE ENTRIES IN numerical & string
            if len(numerical_params | string_params) != (len(numerical_params) + len(string_params)):
                _exc(f"there are duplicate entries in numerical_params and string_params")


            # VALIDATE numerical_params ##############################################################################################

            for _param in numerical_params:

                __ = numerical_params[_param]

                # numerical_params KEYS MUST BE STRINGS
                if not 'str' in str(type(_param)).lower():
                    _exc(f"{_param}: numerical_params' keys must be strings corresponding to args/kwargs of an estimator")

                # numerical_params VALUES MUST BE EAGER LIST-TYPE
                try: __ = compute(__)[0]
                except: pass

                # validate numerical_params' items()[1] are lists
                try: __ = list(__)
                except: _exc(f"{_param}: numerical_params' values must be a list-type, or convertible into a list-type")

                # validate soft/hard/fixed_int/float PART 1 #############################################################################
                try: __[-1] = __[-1].lower()
                except: _exc(f"{_param}: last position must be a string indicating soft/hard/fixed_int/float")

                if not __[-1] in ["hard_integer", "soft_integer", "fixed_integer", "hard_float", "soft_float", "fixed_float"]:
                    _exc(f'numerical_params {_param}: type must be in ["hard_integer", "soft_integer", "fixed_integer", "hard_float", '
                                                        f'"soft_float", "fixed_float"]')
                # END validate soft/hard/fixed_int/float PART 1 #############################################################################

                # VALIDATE POINTS ###########################################################################################################
                try: len(__[-2])  # IF IS A SINGLE NUMBER, CONVERT TO LIST
                except: __[-2] = [__[-2] for _ in range(self.total_passes)]

                _ = NotImplementedError(
                    f'{_param}: "points" must be (i) an integer >= 1 or (ii) a list-type of integers >=1  with len==passes \n\npoints:\n{__[-2]}'
                )

                try: __[-2]['0'] = ''; raise _  # TRY TO TREAT LIKE A DICT
                except NotImplementedError: raise _  # IF SUCCESSFULLY ADDED A KEY
                except TypeError: pass  # FOR PYTHON LIST, MEANS IS NOT A dict, WHICH IS GOOD
                except IndexError: pass  # FOR NUMPY, MEANS IS NOT A dict, WHICH IS GOOD
                except: raise Exception(f'{_param}: Logic testing for dict-like of "points" failed')

                # IF A FLOAT IS IN points
                if True in map(lambda x: 'float' in x, list(map(lambda x: str(type(x)).lower(), __[-2]))): raise _

                # IF ANY NUMBER IN points IS LESS THAN 1
                if np.array(__[-2]).min() < 1: raise _

                if 'soft' in __[-1] and 2 in __[-2]:
                    raise Exception(f'{_param}: Grids of size 2 are not allowed for "soft" numerical params')

                # IF NUMBER OF POINTS IS EVER SET TO 1, ALL SUBSEQUENT POINTS MUST BE 1
                if len(__[-2]) > 1:
                    for idx in range(len(__[-2][:-1])):
                        if __[-2][idx] == 1 and __[-2][idx + 1] > 1:
                            _exc(f"{_param}: ONCE NUMBER OF POINTS IS SET TO 1, ALL SUBSEQUENT POINTS MUST BE 1 \n\npoints={__[-2]}")

                # NUMBER OF POINTS IN points MUST MATCH NUMBER OF PASSES
                if len(__[-2]) != self.total_passes: raise _

                # IF FIXED int/float, NUMBER OF POINTS MUST BE SAME AS ORIGINAL len(GRID), OR 1
                if 'fixed' in __[-2]:
                    for _point in __[-2]:
                        if _point not in [1, __[-2][0]]:
                            _exc(f"if fixed int/float, number of points must be same as first grid or 1 \n\npoints = {__[-2]}")

                del _
                # END VALIDATE POINTS ###########################################################################################################

                # numerical_params' list must meet shape requirements
                if len(__) not in [3,5]:
                    _exc(f"{_param}: numerical_params' values must be a list-type of len 3 or 5 --- \n"
                         f"[[*first_grid], [*number_of_points_for_each_grid], soft/hard/fixed_int/float] - or - \n"
                         f"[search space type, start value, end value, [*number of points for each grid], soft/hard/fixed_int/float]")

                if len(__) == 5:
                    # validate list contains 'linspace/logspace', start_value, end_value ##########################################
                    _msg = f'{_param}: for numerical_params "5" format, first position must be a string: "linspace" or "logspace"'
                    try: numerical_params[_param][0] = numerical_params[_param][0].lower()
                    except: _exc(_msg)
                    if not __[0] in ['linspace', 'logspace']: _exc(_msg)
                    del _msg

                    # start_value / end_value MUST BE NUMERIC
                    for idx in [1,2]:
                        try: np.float64(__[idx])
                        except: _exc(f"{_param}: for '5' format, start_value & end_value must be numeric")

                    # BEAR Y THIS MUST BE INT??
                    if __[0] == 'logspace':
                        for idx, posn in enumerate(['start', 'end'], 1):
                            if not 'int' in str(type(__[idx])).lower():
                                _exc(f'{_param}: {posn}_value must be an integer ({__[idx]}) for logspace')
                    # END validate list contains 'linspace/logspace', start_value, end_value #####################################

                    if __[0] == 'logspace': _grid = np.sort(np.logspace(__[1], __[2], __[3][0]))
                    elif __[0] == 'linspace': _grid = np.sort(np.linspace(__[1], __[2], __[3][0]))

                    __ = [_grid, __[-2], __[-1]]
                    del _grid

                # LEN MUST BE 3

                # validate list contains [first_grid], [number_of_points] in [0,1] slots ###################################
                try: __[0] = np.sort(__[0])
                except: _exc(f"{_param}: for '3' format, first element of list must be a list-type")

                if len(__[0]) != __[1][0]: _exc(f"{_param}: first number_of_points must match length of first grid")
                # END validate list contains first_grid, number_of_points in [0,1] slots ###################################

                # validate soft/hard/fixed_int/float PART 2 #############################################################################
                if 'integer' in __[-1]:
                    # BEAR THINK ABOUT IF HARD OR FIXED INTEGERS CAN EVER BE NEGATIVE (THEY BYPASS SHIFT AND GO STR8 TO DIGGER)
                    # AKA CAN DIGGER HANDLE NEGATIVE INTEGERS
                    if False in [int(i)==i for i in __[0]]:
                        _exc(f"{_param}: when numerical is integer (soft, hard, or fixed), all search values must be integers: \n\ngrid = {__[0]}")
                    if __[-1] == 'soft_integer' and (np.array(__[0]) < 1).any():
                        _exc(f"{_param}: when numerical is soft integer, all search values must be >= 1: \n\ngrid = {__[0]}")
                elif 'float' in __[-1]:
                    # BEAR THINK ABOUT IF HARD OR FIXED FLOATS CAN EVER BE NEGATIVE (THEY BYPASS SHIFT AND GO STR8 TO DIGGER)
                    # AKA CAN DIGGER HANDLE NEGATIVE FLOATS
                    if __[-1] == 'soft_float' and (np.array(__[0]) < 0).any():
                        _exc(f"{_param}: when numerical is soft float, all search values must >= 0: \n\ngrid = {__[0]}")

                # 24_02_07_11_31_00 UNIFORM GAP VALIDATION --- DELETE IF OK
                # if len(__[0]) > 1 and "soft" in __[-1]:
                #     if len(np.unique(np.array(__[0])[1:] - np.array(__[0])[:-1])) != 1:
                #         if len(np.unique(np.log10(__[0])[1:] - np.log10(__[0])[:-1])) != 1:
                #             raise NotImplementedError(f"soft numerics cannot take non-uniform grids")

                # CURRENTLY ONLY HANDLES LOGSPACE BASE 10 OR GREATER
                if len(__[0]) >= 3:
                    log_grid = np.log10(__[0])
                    log_gaps = log_grid[1:] - log_grid[:-1]
                    if len(np.unique(log_gaps)) == 1 and log_gaps[0] < 1:
                        raise NotImplementedError(f"currently only handles logspaces with base 10 or greater")
                    del log_grid, log_gaps
                # END validate soft/hard/fixed_int/float PART 2 #########################################################################


                # BECAUSE CHANGES MAY HAVE BEEN MADE TO THE numerical_params' FORMAT, OVERWRITE THE ORIGINAL ENTRY
                numerical_params[_param] = __

                del __

            self.numerical_params = numerical_params
            # END VALIDATE numerical_params ##########################################################################################

            # VALIDATE string_params ##############################################################################################
            for _param in string_params:

                # validate string_params' dict values are lists that contains (i) a list of str values (ii) optionally a number ######
                if not 'str' in str(type(_param)).lower():
                    _exc(f"string_params' keys must be strings corresponding to args/kwargs of an estimator")

                try: string_params[_param] = compute(string_params[_param])[0]
                except: pass

                # validate string_params' items[1] are list-types
                try: string_params[_param] = list(string_params[_param])
                except: _exc(f"string_params' values must be a list-type")

                __ = string_params[_param]

                # FORMAT STANDARDIZATION #########################################################################################
                if len(__)==0: _exc(f"string_params' values cannot be an empty list")
                elif len(__)==1:
                    try:
                        # IF THE INNER OBJECT IS A LIST, THEN WAS PASSED AS [[*stuff]], SO MAKE [[*stuff], a_big_numeric]
                        __ = [list(__[0]), 1_000_000]
                    except:
                        # IF THE INNER OBJECT IS NOT A LIST, THEN WAS (POSSIBLY) PASSED AS [one_thing], SO MAKE [[one_thing], a_big_numeric]
                        __ = [__, 1_000_000]

                elif len(__)==2:
                    # IF THE FIRST OBJECT IS A LIST-TYPE, WAS (POSSIBLY) PASSED AS [[*stuff], numeric]
                    # IF THE FIRST OBJECT IS NOT A LIST, THEN WAS (POSSIBLY) PASSED AS [*two_things], SO MAKE [[*two_things], a_big_numeric]
                    try: __[0] = list(__[0])
                    except: __ = [__, 1_000_000]

                elif len(__) > 2:
                    # THIS MUST HAVE BEEN PASSED AS [*stuff], SO MAKE [[*stuff], a_big_numeric]
                    if 'str' not in str(type(__[-1])).lower():
                        _exc(f"string_params must be passed as [*params], or [[*params], integer]")
                    __ = [__, 1_000_000]
                # END FORMAT STANDARDIZATION ####################################################################################

                if len(__) != 2:
                    _exc(f"string_params' value standardizer logic is failing, something got thru with len != 2")

                if 'list' not in str(type(__[0])):
                    _exc(f"string_params' value standardizer logic is failing, first object is not a list")

                if 'int' not in str(type(__[1])).lower() or __[1] < 1:
                    _exc(f"if passing string_params as [list((kw)args), pass_number], pass_number must be an integer >= 1")

                if False in map(lambda x: 'str' in str(type(x)).lower(), __[0]):
                    _exc(f"string_params' (kw)args must be a list of strings")
                # END validate string_params' dict values are lists that contain a list of str values & number to consider ######

                string_params[_param] = __
                del __

            self.string_params = string_params
            # END VALIDATE string_params ##########################################################################################

            del _exc, numerical_params, string_params

            # END init VALIDATION ###########################################################################################################

            # super() instantiated in init() for access to XGridSearchCV's pre-run attrs and methods
            super().__init__(estimator, {}, **parent_gscv_kwargs)

            # THIS MUST STAY HERE FOR demo TO WORK
            self.reset()

        # END __init__() ##################################################################################################################
        ###################################################################################################################################
        ###################################################################################################################################


        # reset() #######################################################################################################################
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

            # IS_LOGSPACE IS DYNAMIC, WILL CHANGE WHEN A PARAM'S SEARCH GRID TRANSITIONS FROM LOGSPACE TO LINSPACE
            self.IS_LOGSPACE = dict()
            for _param in self.numerical_params | self.string_params:
                if _param in self.numerical_params:
                    __ = self.numerical_params[_param]
                    # "soft" & "hard" CAN BE LOGSPACES, BUT "fixed" CANNOT
                    if "fixed" in __[-1]: self.IS_LOGSPACE[_param] = False; continue
                    if 0 in __[0]: self.IS_LOGSPACE[_param] = False; continue
                    if __[-2][0] <= 2: self.IS_LOGSPACE[_param] = False; continue
                    # IF IS LOGSPACE, PUT IN THE SIZE OF THE GAP (bool(>0) WILL RETURN True)
                    log_gap = np.log10(__[0])[1:]-np.log10(__[0])[:-1]
                    if len(np.unique(log_gap))==1:   # UNIFORM GAP SIZE IN LOG SCALE
                        if 'hard' in __[-1] and log_gap[0] > 1:
                            raise Exception(f'"hard" logspaces must have unit intervals or smaller (intervals cannot be > 1)')
                        else: self.IS_LOGSPACE[_param] = log_gap[0]
                    else: self.IS_LOGSPACE[_param] = False   # MUST BE LINSPACE
                    del __, log_gap
                elif _param in self.string_params: self.IS_LOGSPACE[_param] = False

            # ONLY POPULATE WITH numerical_params WITH "soft" BOUNDARY AND START AS FALSE
            self.PARAM_HAS_LANDED_INSIDE_THE_EDGES = \
                {hprmtr: False for hprmtr in self.numerical_params if 'soft' in self.numerical_params[hprmtr][-1]}

        # END reset() #############################################################################################################


        ###################################################################################################################################
        # FUNCTIONAL METHOD ###############################################################################################################
        def get_next_param_grid(self, _pass, best_params_from_previous_pass):

            """
            :param _pass: int - iteration counter
            :param best_params_from_previous_pass: dict - best_params_ returned by Gridsearch for the previous pass
            :return: dict of grids to be passed to GridSearchCV for the next pass, must be in param_grid format

             Core functional method. On first pass (pass zero), populate GRIDS with the search grids as specified in
             numerical_params and string_params. For subsequent passes, generate new grids based on the previous grid (as
             held in GRIDS) and its associated best_params_ returned from GridSearchCV.
            """

            # SEED best_params_from_previous_pass WITH None ON FIRST PASS

            #####################################################################################################################
            # FUNCTIONAL METHOD VALIDATION ######################################################################################
            # ABBREVIATED EXCEPTION HANDLING
            def _exc(text):
                raise Exception(f'{text}')

            # best_params_ from dask/sklearn.GridSearchCV looks like
            # {'C': 1, 'l1_ratio': 0.9}
            if len(self.GRIDS) != 0:
                if 'dict' not in str(type(best_params_from_previous_pass)):
                    _exc(f'best_params_from_previous_pass is not a dict. Has GridSearchCV best_params_ output changed?')
                if len(best_params_from_previous_pass) != len(self.GRIDS[_pass-1]):
                    _exc(f'len(best_params_from_previous_pass) ({len(best_params_from_previous_pass)}) != '
                         f'len(params) from previous pass ({len(self.GRIDS[_pass])})')
                for param_ in best_params_from_previous_pass:
                    # VALIDATE best_param_ KEYS WERE IN ITS GRID
                    if param_ not in self.GRIDS[_pass-1]:
                        _exc(f'{param_} in best_params_from_previous_pass is not in given params from the previous pass')

                    # VALIDATE THAT RETURNED best_params_ HAS VALUES THAT ARE WITHIN THE PREVIOUS SEARCH SPACE
                    if best_params_from_previous_pass[param_] not in self.GRIDS[_pass-1][param_]:
                        _exc(f"best_params_ contains a value that was not in its given search space")

                for param_ in self.GRIDS[_pass-1]:
                    # VALIDATE GRID KEYS WERE ARE IN best_params_from_previous_pass
                    if param_ not in best_params_from_previous_pass:
                        _exc(f'{param_} in GRIDS[{_pass}] is not in best_params_from_previous_pass')

            # END FUNCTIONAL METHOD VALIDATION #################################################################################
            #####################################################################################################################

            # AT THIS POINT SHOULD HAVE VALID best_params_from_previous_pass WITH BEST VALUES FOR EACH.
            # FEED EACH INTO AN ALGORITHM THAT BUILDS ITS NEXT GRID

            #####################################################################################################################
            # FUNCTIONAL METHOD OPERATION ######################################################################################

            # IF ON FIRST PASS, THE FIRST SET OF GRIDS MUST BE FILLED FROM numerical/string_params #########################
            if len(self.GRIDS) == 0:                 # best_params_from_previous_pass is None

                # READ numerical_params & string_params TO POPULATE GRIDS[0]
                self.GRIDS[_pass] = {n_param: self.numerical_params[n_param][0] for n_param in self.numerical_params} | \
                                {s_param: self.string_params[s_param][0] for s_param in self.string_params}

                # self.IS_LOGSPACE does not change
                return self.GRIDS[_pass]
            # END FIRST PASS, THE FIRST SET OF GRIDS MUST BE FILLED FROM numerical/string_params ###########################
            else:
                self.GRIDS[_pass] = dict()

            # IF ON A SUBSEQUENT PASS, MUST USE best_params_from_previous_pass AND THE LAST GRID USED TO BUILD A NEW GRID ####################

            #    v v v   elif len(self.GRIDS) > 0   v v v

            # must distinguish if a soft num param has fallen inside the edges of its grid at one point or not.
            # A string_parameter AND hard/fixed numerical_parameter CANNOT BE ON AN EDGE (ALGORITHMICALLY SPEAKING)!
            # this is not needed after the pass where all soft num fall inside the edges (all values in PARAM_HAS_LANDED_INSIDE_THE_EDGES
            # will be False and cannot gain re-entry to the place where they could be set back to True)
            # UPDATE PARAM_HAS_LANDED_INSIDE_THE_EDGES & ITS EVENT VARIABLES #########################################################
            if False in self.PARAM_HAS_LANDED_INSIDE_THE_EDGES.values() and self.shift_ctr < self.max_shifts:
                # THE ONLY PARAMS THAT POPULATE PARAM_HAS_LANDED_INSIDE_THE_EDGES ARE SOFT LIN/LOGSPACE & HARD LOGSPACE
                # IF SOFT LANDED INSIDE THE EDGES, THEN TRULY LANDED "INSIDE THE EDGES" AND WONT BE SHIFTED (True)
                # ANY LOGSPACE WITH > 1 INTERVAL THEN NOT YET "INSIDE THE EDGES" (False)
                # IF HARD LOGSPACE WITH GAP <= 1, THEN ALWAYS INSIDE THE EDGES
                # IF ONLY ONE OPTION IN ITS GRID, CANNOT BE SHIFTED (True)
                # IF LANDED ON AN EDGE, BUT THAT EDGE IS A UNIVERSAL HARD BOUND (0 FOR float, 1 FOR int) THEN WONT BE SHIFTED (True)
                # IF LANDED ON AN EDGE, BUT THAT EDGE IS NOT A UNIV-HARD BOUND, USER-HARD BOUND, OR FIXED, THEN SHIFT (STAYS OR RE-BECOMES False)
                for _param in self.PARAM_HAS_LANDED_INSIDE_THE_EDGES:
                    _best, _grid, _holder = best_params_from_previous_pass[_param], self.GRIDS[_pass-1][_param], None

                    if self.IS_LOGSPACE[_param] > 1:
                        _holder = False
                    elif not np.isclose(_best,  _grid.min(), rtol=1e-6) and not np.isclose(_best, _grid.max(), rtol=1e-6):
                        _holder = True
                    else:   # MUST BE ON AN EDGE
                        __ = self.numerical_params[_param]
                        if __[1][_pass] == 1: _holder = True        # SHOULD BE EQUIVALENT TO len(_grid)==1
                        elif __[-1]=='soft_integer' and _best == 1: _holder = True
                        elif __[-1]=='soft_float' and _best == 0: _holder = True
                        else: _holder = False

                    self.PARAM_HAS_LANDED_INSIDE_THE_EDGES[_param] = _holder

                try: del _best, _grid, _holder
                except: pass
                try: del __
                except: pass
            # END UPDATE PARAM_HAS_LANDED_INSIDE_THE_EDGES & ITS EVENT VARIABLES ########################################################

            # IF ANY OF THE PARAMS ARE STILL LANDING ON THE EDGES, MUST SLIDE THE GRID FOR THOSE PARAMS AND RERUN ALL THE
            # OTHER PARAMS WITH THEIR SAME GRID ############################################################################
            if False in self.PARAM_HAS_LANDED_INSIDE_THE_EDGES.values() and self.shift_ctr < self.max_shifts:

                self.shift_ctr += 1

                for _param in self.numerical_params | self.string_params:
                    OLD_GRID = np.array(self.GRIDS[_pass-1][_param])
                    OLD_GRID.sort()
                    _best = best_params_from_previous_pass[_param]
                    if self.PARAM_HAS_LANDED_INSIDE_THE_EDGES.get(_param, True) == True:
                        # IF WAS A STRING, HARD_NUM, FIXED_NUM, OR A NUM INSIDE THE EDGES, CARRY ITS LAST GRID OVER TO NEW PASS
                        self.GRIDS[_pass][_param] = OLD_GRID
                    elif self.PARAM_HAS_LANDED_INSIDE_THE_EDGES.get(_param, True) == False:
                        # linspace/logspace, WHICH EDGE IT LANDED ON (IF ANY), NUMBER OF POINTS, LOG GAP MATTERS HERE ###########
                        # ALREADY KNOW IF linspace/logspace FROM IS_LOGSPACE

                        # GET WHICH EDGE _best LANDED ON (IF ANY)
                        _left, _right = np.isclose(_best, OLD_GRID.min(), rtol=1e-6), np.isclose(_best, OLD_GRID.max(), rtol=1e-6)

                        # GET POINTS (FOR ALL soft LINSPACE AND LOGSPACE WHERE GAP <=1; LOGSPACE WHERE GAP > 1 HANDLED BELOW)
                        _points = self.numerical_params[_param][1][_pass - 1]
                        # END linspace/logspace, UNIFORM SPACE, WHICH EDGE IT LANDED ON (IF ANY), NUMBER OF POINTS ######################

                        # REMEMBER THAT IF (ON AN EDGE) ITS IN HERE TO BE SHIFTED; IF (NOT ON AN EDGE
                        # LOG GAP MUST BE > 1) ITS IN HERE TO BE REGAPPED

                        if not self.IS_LOGSPACE[_param]:
                            # THE ONLY THING THAT CAN HAPPEN HERE IS SHIFT

                            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                            # NEW WAY --- _offset ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                            # left shift offset = (second lowest number) - max
                            # right shift offset = (second highest number) - min

                            if _left: _offset = OLD_GRID[1] - OLD_GRID.max()
                            elif _right: _offset = OLD_GRID[-2] - OLD_GRID.min()

                            OLD_GRID += _offset
                            del _offset

                            if 'float' in self.numerical_params[_param][-1] and (OLD_GRID < 0).any():
                                OLD_GRID += np.abs(OLD_GRID.min())
                            elif 'integer' in self.numerical_params[_param][-1] and (OLD_GRID < 1).any():
                                OLD_GRID += (np.abs(OLD_GRID.min()) + 1)

                            self.GRIDS[_pass][_param] = OLD_GRID
                            # END NEW WAY --- _offset ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                            # OLD WAY --- _span ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                            # BEAR 24_02_07_09_34_00 DELETE IF OK
                            # if _left: _new_right = OLD_GRID[1]
                            # elif _right: _new_left = OLD_GRID[-2]
                            #
                            # _span = np.subtract(*OLD_GRID[[-1, 0]])
                            #
                            # if _left: _new_left = _new_right - _span
                            # if _right: _new_right = _new_left + _span
                            #
                            # if 'float' in self.numerical_params[_param][-1] and _new_left < 0:
                            #     _new_left, _new_right = 0, _span
                            # elif 'integer' in self.numerical_params[_param][-1] and _new_left < 1:
                            #     _new_left, _new_right = 1, 1 + _span
                            #
                            # self.GRIDS[_pass][_param] = np.linspace(_new_left, _new_right, _points)
                            # END OLD WAY --- _span ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                        elif self.IS_LOGSPACE[_param]:

                            OLD_GRID = np.log10(OLD_GRID)

                            # IN HERE FOR SHIFT ###############################################################################
                            if (_left or _right):
                                # left shift offset = (second lowest number) - max
                                # right shift offset = (second highest number) - min

                                if _left: _offset = OLD_GRID[1] - OLD_GRID.max()
                                elif _right:_offset = OLD_GRID[-2] - OLD_GRID.min()

                                OLD_GRID += _offset
                                del _offset
                            # END IN HERE FOR SHIFT ###########################################################################

                            # LOGSPACE GAP > 1 #################################################################################
                            elif not (_left or _right):  # IN HERE FOR REGAP
                                if not self.IS_LOGSPACE[_param] > 1:
                                    raise _exc(f"logspace grid has entered regap code section but gap is already <= 1")
                                # MUST BE OFF AN EDGE HERE OR WOULD HAVE TRIGGERED SHIFTER CODE FIRST

                                # IF THIS IS THE FIRST PASS AFTER ALL PARAMS FELL INSIDE THE EDGES OF THEIR RANGES AND YOU HAVE REACHED THIS POINT,
                                # INTERVAL IS > 1, REGAP THAT DOWN TO 1. USE THE VALUES THAT FALL TO THE LEFT AND RIGHT OF THE BEST VALUE
                                # TO CREATE A NEW RANGE WITH INCREMENT 1.

                                posn_of_best = np.isclose(sorted(self.GRIDS[_pass - 1][_param]), _best,rtol=1e-6)

                                assert posn_of_best.astype(bool).sum() == 1, (
                                    f"locating position of best param in its grid is failing. "
                                    f"expected one positional match in grid, found {posn_of_best.astype(bool).sum()}.")

                                posn_of_best = np.arange(len(self.GRIDS[_pass - 1][_param]))[posn_of_best][0]

                                _ = self.GRIDS[_pass - 1][_param][posn_of_best - 1]
                                if _ <= _best: _new_left = np.floor(np.log10(_))
                                elif _ > _best: _new_left = np.ceil(np.log10(_))

                                _ = self.GRIDS[_pass - 1][_param][posn_of_best + 1]
                                if _ >= _best: _new_right = np.ceil(np.log10(_))
                                elif _ < _best: _new_right = np.floor(np.log10(_))

                                del _

                                _points = abs(_new_right - _new_left) + 1

                                OLD_GRID = np.linspace(_new_left, _new_right, _points)  # DOING 10** BELOW

                                del posn_of_best, _new_left, _new_right
                            # END LOGSPACE GAP > 1 #################################################################################

                            if 'integer' in self.numerical_params[_param][-1] and (OLD_GRID < 0).any():
                                OLD_GRID += np.abs(OLD_GRID.min())

                            # DONT TO CHECK FOR 'float', LOGSPACE WILL ALWAYS BE > 0

                            self.GRIDS[_pass][_param] = 10 ** OLD_GRID

                        del _left, _right, _points

                    else: _exc(f"A non-boolean value is in PARAM_HAS_LANDED_INSIDE_THE_EDGES")

                del OLD_GRID, _best

                # WE ARE IN THIS CONDITIONAL BECAUSE AT LEAST 1 False WAS in self.PARAM_HAS_LANDED_INSIDE_THE_EDGES.values()
                # THAT MEANS AT LEAST 1 NUMERICAL WAS ON AN EDGE OR A LOGSPACE HAD GAP > 1, WHICH MEANS THE NEXT COMING PASS
                # WILL HAVE IDENTICAL # POINTS FOR ALL (STRING) & (NUMERICAL PARAMS EXCEPT FOR WHERE LOGSPACE GAP > 1 AND FELL
                # INSIDE THE EDGES)
                # FOR NUMERICAL PARAMS:
                # REPLICATE THE PREVIOUS POINTS INTO THE CURRENT PASS AND PUSH THE NEXT VALUES OVER TO THE RIGHT; IF
                # total_passes IS HARD, DROP LAST VALUE: EG, [10, 5, 4, 3] ON EDGE AT PASS 0, BECOMES [10, 10, 5, 4] ON PASS 1
                # FOR PARAM WHERE LOGSPACE GAP > 1, OVERWRITE THAT PARAM'S COMING PASS'S _points WITH POINTS FOR GAP==1

                if not self.total_passes_is_hard:
                    self.total_passes += 1

                for _param in self.numerical_params:
                    self.numerical_params[_param][1].insert(_pass, self.numerical_params[_param][1][_pass-1])
                    if self.total_passes_is_hard:
                        self.numerical_params[_param][1] = self.numerical_params[_param][1][:-1]
                    if self.IS_LOGSPACE[_param] > 1: # self.IS_LOGSPACE ONLY CHANGES WHERE GAP > 1
                        # GRIDS SHOULD BE UPDATED WITH THE NEXT GRID A FEW LINES ABOVE, SO USE THAT TO GET NEW _points
                        self.numerical_params[_param][1][_pass] = len(self.GRIDS[_pass][_param])
                        # SINCE False IN INSIDE_THE_EDGES DOESNT INDICATE SHIFT OR REGAP, REMEASURE GAP TO KNOW IF WAS REGAPPED
                        _grid = self.GRIDS[_pass][_param]
                        log_gap = _grid[1:] - _grid[:-1]
                        if len(np.unique(log_gap))==1 and log_gap[0]==1:
                            self.IS_LOGSPACE[_param] = 1
                        del _grid, log_gap

                # FOR STRING PARAMS, INCREMENT THE pass_on_which_to_use_only_the_single_best_value VALUE
                for _param in self.string_params:
                    self.string_params[_param][1] += 1

                return self.GRIDS[_pass]
                # END SLIDE GRIDS ##############################################################################################

            # END STILL ON THE EDGES #######################################################################################



            # IF REACHED THIS POINT, (i) THIS IS NOT FIRST PASS (ii) EVERYTHING IN PARAM_HAS_LANDED_INSIDE_THE_EDGES IS True,
            # (iii) ANY LOGSPACES HAVE AN INTERVAL OF <= 1 AND WILL CONVERT TO LINSPACES

            for hprmtr in self.numerical_params:

                _type = self.numerical_params[hprmtr][-1]

                _points = self.numerical_params[hprmtr][1][_pass]
                _best = best_params_from_previous_pass[hprmtr]
                _grid = self.GRIDS[_pass-1][hprmtr]

                best_param_posn = np.isclose(_grid, _best, rtol=1e-6)

                if best_param_posn.sum() != 1:
                    _exc(f"uniquely locating best_param position in search grid is failing, should locate to 1 position,"
                         f"but locating to {best_param_posn.sum()}")

                best_param_posn = np.arange(len(_grid))[best_param_posn][0]

                # ONLY NEEDED FOR 'hard' NUMERICAL
                hard_min = self.GRIDS[0][hprmtr][0]
                hard_max = self.GRIDS[0][hprmtr][-1]

                if _points == 1:
                    self.GRIDS[_pass][hprmtr] = [_best]
                    self.IS_LOGSPACE[hprmtr] = False   # MAY HAVE ALREADY BEEN FALSE
                    continue

                elif 'fixed' in self.numerical_params[hprmtr][-1]:
                    # THIS MUST BE AFTER _points == 1
                    self.GRIDS[_pass][hprmtr] = self.numerical_params[hprmtr][0]
                    continue


                if 'HARD' in _type.upper(): is_hard = True
                elif 'SOFT' in _type.upper(): is_hard = False
                else: raise Exception(f"{hprmtr}: bound_type must contain 'hard' or 'soft' ({_type})")


                if 'integer' in _type:

                    # RULES SURROUNDING LOGPACES AND INTEGERS
                    # --- CAN ONLY BE A "LOGSPACE" IF len(GRID) >= 3
                    # --- CAN ONLY BE A "LOGSPACE" IF log10 GAPS ARE UNIFORM (24_02_08_15_39_00 AND == 1)
                    # --- "FIXED" IN THIS ALGORITHM CANNOT BE LOGSPACE
                    # --- LOGSPACE IS POSITIVE DEFINITE IN LINEAR SPACE
                    # --- IF A HARD LOGSPACE, log10 GAPS MUST BE <= 1 AT __init__ TIME (ENFORCED BY VALIDATION)
                    # --- IF A SOFT LOGSPACE, REGAP CODE FORCES log10 GAPS TO <= 1
                    # --- INTEGER VALUES IN LINEAR SPACE IS ENFORCED BY VALIDATION IN __init__
                    # --- IF LINEAR GAP == 1, CANT DRILL ANY DEEPER

                    def new_integer_grid_mapper(location, _left, _right, _points):

                        _gap = int(_right - _left)
                        if _gap == 0:
                            raise Exception(f"_gap == 0")
                        elif _gap == 1:
                            _new_points = 2
                            POINTS_DICT = {2: np.linspace(_left, _right, 2)}
                        elif _gap == 2:
                            _new_points = 3
                            POINTS_DICT = {3: np.linspace(_left, _right, 3)}
                        else:
                            POINTS_DICT = dict()
                            NEW_LINSPACE, LAST_LINSPACE = [], []
                            # _points CANNOT BE 0, 1
                            for test_points in range(2, _gap + 1):

                                if location == 'LEFT':
                                    new_right = _left + test_points * np.ceil(_gap / test_points)
                                    NEW_LINSPACE = np.linspace(_left, new_right, test_points + 1, dtype=np.int32)
                                    del new_right
                                    NEW_LINSPACE = NEW_LINSPACE[NEW_LINSPACE < _right]
                                elif location == 'RIGHT':
                                    new_left = _right - test_points * np.ceil(_gap / test_points)
                                    NEW_LINSPACE = np.linspace(new_left, _right, test_points + 1, dtype=np.int32)
                                    del new_left
                                    NEW_LINSPACE = NEW_LINSPACE[NEW_LINSPACE > _left]
                                elif location == 'MIDDLE':
                                    new_right = _left + test_points * np.ceil(_gap / test_points)
                                    NEW_LINSPACE = np.linspace(_left, new_right, test_points + 2, dtype=np.int32)
                                    NEW_LINSPACE = NEW_LINSPACE[(NEW_LINSPACE > _left) * (NEW_LINSPACE < _right)]

                                if len(NEW_LINSPACE) == len(LAST_LINSPACE):
                                    continue
                                else:
                                    POINTS_DICT[test_points] = NEW_LINSPACE
                                    LAST_LINSPACE = NEW_LINSPACE.copy()

                                if min(np.abs(np.fromiter(POINTS_DICT.keys(), dtype=np.uint8)-_points)) < (test_points - _points):
                                    break

                            del LAST_LINSPACE, NEW_LINSPACE

                            POINTS_LIST = np.fromiter(POINTS_DICT.keys(), dtype=np.uint8)
                            SPREADS = np.abs(POINTS_LIST - _points)
                            CLOSEST = POINTS_LIST[SPREADS == SPREADS.min()]
                            _new_points = CLOSEST.min()
                            del POINTS_LIST, SPREADS, CLOSEST

                        return _new_points, POINTS_DICT[_new_points]


                    if self.IS_LOGSPACE[hprmtr]:

                        # THIS CAN ONLY BE ACCESSED ON THE FIRST PASS AFTER SHIFTER

                        if self.IS_LOGSPACE[hprmtr] > 1:
                            _exc(f"{hprmtr}: an integer logspace with log10 gap > 1 has made it into digging section")

                        if best_param_posn == 0:
                            location = 'LEFT'
                            _left, _right = _grid[0], _grid[1]
                            if not is_hard and _grid[0] != 1:
                                raise Exception(f"{hprmtr}: a soft logspace integer is on a left edge and value != 1")
                        elif best_param_posn == len(_grid) - 1:
                            location = 'RIGHT'
                            _left, _right = _grid[0], _grid[-1]
                            if not is_hard:
                                raise Exception(f"{hprmtr}: a soft logspace integer is on a right edge... should have shifted")
                        elif best_param_posn not in range(len(_grid)):
                            raise Exception(f"{hprmtr}: best_param_posn ({best_param_posn}) is not in range of _grid")
                        else:
                            location = 'MIDDLE'
                            if is_hard:
                                _left, _right = _grid[0], _grid[best_param_posn + 1]
                            elif not is_hard:
                                _left, _right = 1, _grid[best_param_posn + 1]

                        _new_points, new_grid = new_integer_grid_mapper(location, _left, _right, _points)

                        self.GRIDS[_pass][hprmtr] = new_grid
                        self.numerical_params[hprmtr][-2][_pass] = _new_points
                        del _new_points, new_grid

                        self.IS_LOGSPACE[hprmtr] = False

                    elif not self.IS_LOGSPACE[hprmtr]:

                        if best_param_posn == 0:
                            location = 'LEFT'
                            _left, _right = _grid[0], _grid[1]
                            intermediate_gap = _right - _left
                            if is_hard:
                                # CANT SAY _pass==0 IF HAD SOME SHIFTING; ADJUST BY SHIFTS
                                if _pass > self.shift_ctr:
                                    _left = max(self.GRIDS[0][hprmtr][0], _left-intermediate_gap)
                                # elif _pass==self.shift_ctr, _left is already the hard edge
                            elif not is_hard:
                                if _pass==self.shift_ctr and not _grid[0] == 1:
                                    raise Exception(f"{hprmtr}: a soft integer is on a left edge after shifter and value != 1")
                                _left = max(1, _left-intermediate_gap)
                            del intermediate_gap
                        elif best_param_posn == len(_grid) - 1:
                            location = 'RIGHT'
                            _left, _right = _grid[-2], _grid[-1]
                            intermediate_gap = _right - _left
                            if is_hard:
                                # CANT SAY _pass==0 IF HAD SOME SHIFTING; ADJUST BY SHIFTS
                                if _pass > self.shift_ctr:
                                    _right = min(self.GRIDS[0][hprmtr][-1], _right+intermediate_gap)
                                # elif _pass==self.shift_ctr, _right is already the hard edge
                            elif not is_hard:
                                if _pass==self.shift_ctr:
                                    raise Exception(f"{hprmtr}: a soft integer is on a right edge... should have shifted")
                                _right = _right + intermediate_gap
                            del intermediate_gap
                        elif best_param_posn not in range(len(_grid)):
                            raise Exception(f"{hprmtr}: best_param_posn ({best_param_posn}) is not in range of _grid")
                        else:
                            location = 'MIDDLE'
                            _left, _right = _grid[best_param_posn - 1], _grid[best_param_posn + 1]

                        _new_points, new_grid = new_integer_grid_mapper(location, _left, _right, _points)

                        self.GRIDS[_pass][hprmtr] = new_grid
                        self.numerical_params[hprmtr][-2][_pass] = _new_points
                        del _new_points, new_grid

                    del _left, _right, new_integer_grid_mapper

                elif 'float' in _type:

                    if self.IS_LOGSPACE[hprmtr]:  # CAN ONLY HAPPEN ON FIRST PASS AFTER SHIFTER

                        if self.IS_LOGSPACE[hprmtr] > 1:
                            _exc(f"{hprmtr}: an integer logspace with log10 gap > 1 has made it into digging section")

                        # CONVERT THE LOGSPACE GRID BACK TO LOGS, THIS SHOULD CREATE EQUAL GAPS BETWEEN ALL THE POINTS
                        log_best = np.log10(_best)

                        match best_param_posn:
                            case 0:  # IF ON THE LEFT EDGE OF GRID
                                if 'hard' in _type and  _grid[0] in [hard_min, hard_max]:
                                    # THIS GIVES CORRECT RANGE BUT DOESNT GUARANTEE NICE DIVISIONS
                                    # BECAUSE _left CANT BE JUST SET TO ZERO BECAUSE OF HARD BOUND
                                    _left, _right = _grid[0], _grid[1]
                                    padded_space = np.linspace(_left, _right, _points+1)
                                    chopped_space = padded_space[:-1]
                                    self.GRIDS[_pass][hprmtr] = chopped_space
                                else:
                                    _left = 0
                                    _right = 10**(log_best + np.abs(self.IS_LOGSPACE[hprmtr]))
                                    padded_space = np.linspace(_left, _right, _points + 2)
                                    chopped_space = padded_space[1:-1]
                                    self.GRIDS[_pass][hprmtr] = chopped_space

                            case best_param_posn if best_param_posn==(len(_grid)-1):   # RIGHT EDGE OF GRID
                                if 'hard' in _type and _grid[-1] in [hard_min, hard_max]:
                                    # THIS GIVES CORRECT RANGE BUT DOESNT GUARANTEE NICE DIVISIONS
                                    # BECAUSE _left CANT BE JUST SET TO ZERO BECAUSE OF HARD BOUND
                                    _left, _right = _grid[-2], _grid[-1]
                                    padded_space = np.linspace(_left, _right, _points+1)
                                    chopped_space = padded_space[1:]
                                    self.GRIDS[_pass][hprmtr] = chopped_space
                                else:
                                    _left = 0
                                    _right = 10**(log_best + np.abs(self.IS_LOGSPACE[hprmtr]))
                                    padded_space = np.linspace(_left, _right, _points + 2)
                                    chopped_space = padded_space[1:-1]
                                    self.GRIDS[_pass][hprmtr] = chopped_space

                            case other:   # SOMEWHERE IN THE MIDDLE OF THE GRID
                                _left = 0
                                _right = 10**(log_best + np.abs(self.IS_LOGSPACE[hprmtr]))
                                padded_space = np.linspace(_left, _right, _points + 2)
                                chopped_space = padded_space[1:-1]
                                self.GRIDS[_pass][hprmtr] = chopped_space

                        del log_best

                        self.IS_LOGSPACE[hprmtr] = False

                    elif not self.IS_LOGSPACE[hprmtr]:
                        _gap = np.subtract(*_grid[[1,0]])

                        match best_param_posn:
                            case 0:   # IF ON THE LEFT EDGE OF THE GRID
                                if 'hard' in _type:
                                    _left, _right = _grid[0], _grid[1]
                                    if _grid[0] in [hard_min, hard_max]:
                                        padded_space = np.linspace(_left, _right, _points + 2)
                                        chopped_space = padded_space[:-1]
                                        _points += 1
                                        self.GRIDS[_pass][hprmtr] = chopped_space
                                    else:
                                        _right = _grid[1]
                                        _left = max(self.GRIDS[0][hprmtr][0], _left - (_right-_left))
                                        padded_space = np.linspace(_left, _right, _points + 1)
                                        chopped_space = padded_space[:-1]
                                        self.GRIDS[_pass][hprmtr] = chopped_space
                                else:
                                    _left, _right = max(_grid[0]-_gap, 0), _grid[1]
                                    padded_space = np.linspace(_left, _right, _points + 2)
                                    chopped_space = padded_space[1:-1]
                                    self.GRIDS[_pass][hprmtr] = chopped_space

                            case best_param_posn if best_param_posn==(len(_grid)-1): # IF ON THE RIGHT EDGE OF THE GRID
                                if 'hard' in _type and _grid[-1] in [hard_min, hard_max]:
                                    _left, _right = _grid[-2], _grid[-1]
                                    padded_space = np.linspace(_left, _right, _points + 2)
                                    chopped_space = padded_space[1:]
                                    _points += 1
                                    self.GRIDS[_pass][hprmtr] = chopped_space
                                else:
                                    _left, _right = _grid[-2], _grid[-1]+_gap
                                    padded_space = np.linspace(_left, _right, _points + 2)
                                    chopped_space = padded_space[1:-1]
                                    self.GRIDS[_pass][hprmtr] = chopped_space

                            case other:     # SOMEWHERE IN THE MIDDLE OF THE GRID
                                _left, _right = _grid[best_param_posn-1], _grid[best_param_posn+1]
                                padded_space = np.linspace(_left, _right, _points + 2)
                                chopped_space = padded_space[1:-1]
                                self.GRIDS[_pass][hprmtr] = chopped_space

                        del _gap

                    del _left, _right, padded_space, chopped_space

                else: _exc(f"logic for selecting hard/soft_int/float to fill the next grid is failing")

                # IF ANY ADJUSTMENTS WERE MADE TO _points, CAPTURE IN numerical_params
                self.numerical_params[hprmtr][1][_pass] = _points


            try: del _best, _grid, _points, _type, is_hard, best_param_posn, hard_min, hard_max
            except: pass

            for hprmtr in self.string_params:

                _spr = self.string_params[hprmtr]

                if _pass >= _spr[1]:
                    self.GRIDS[_pass][hprmtr] = [best_params_from_previous_pass[hprmtr]]
                else:
                    self.GRIDS[_pass][hprmtr] = _spr[0]

            try: del _spr
            except: pass

            del _exc
            return self.GRIDS[_pass]
            # END IF ON A SUBSEQUENT PASS, MUST USE best_params_from_previous_pass AND THE LAST GRID USED TO BUILD A NEW GRID #####################

            # END FUNCTIONAL METHOD OPERATION ###################################################################################
            #####################################################################################################################

        # END FUNCTIONAL METHOD ########################################################################################################
        ################################################################################################################################


        def demo(self, true_best_params=None):

            """
            :param true_best_params: dict - dict of user-generated true best parameters for the given numerical / string params.
                                     If not passed, random best params are generated.
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
            best_params_from_previous_pass = None
            while _pass < self.total_passes:

                _param_grid = self.get_next_param_grid(_pass, best_params_from_previous_pass)

                if self.agscv_verbose:
                    print(f"\nPass {_pass+1} starting grids:")
                    for k, v in _param_grid.items():
                        try: print(f'{k}'.ljust(15), np.round(v, 4))
                        except: print(f'{k}'.ljust(15), v)

                # SHOULD GridSearchCV param_grid FORMAT EVER CHANGE, CODE WOULD GO HERE TO ADAPT THE get_next_param_grid
                # OUTPUT TO THE REQUIRED XGridSearchCV.param_grid FORMAT
                adapted_param_grid = _param_grid

                # (param_grid/param_distributions/parameters is overwritten and X_GSCV is fit()) total_passes times.
                # After a run of AutoGSCV, the final results held in the final pass attrs and methods of X_GSCV are
                # exposed by the AutoGridSearch instance.
                try: self.param_grid = adapted_param_grid
                except: pass
                try: self.param_distributions = adapted_param_grid
                except: pass
                try: self.parameters = adapted_param_grid
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
                best_params_from_previous_pass = adapted_best_params  # GOES BACK INTO self.get_next_param_grid
                del adapted_best_params

                _pass += 1

            del best_params_from_previous_pass

            if self.agscv_verbose:
                print(f"\nfit successfully completed {self.total_passes} pass(es) with {self.shift_ctr} shift pass(es).")


            return self


    return AutoGridSearch










if __name__ == '__main__':
    pass















































