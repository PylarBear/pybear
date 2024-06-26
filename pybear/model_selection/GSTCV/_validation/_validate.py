# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from _estimator import _val_estimator
from _n_jobs import _val_n_jobs
from _verbose import _val_verbose


def _validate(

    _estimator,







    ) -> []:


    def _exc(reason):
        raise Exception(f"{reason}")


    # VALIDATE estimator ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

    __is_dask_estimator = _val_estimator(_estimator)

    # END VALIDATE estimator ** ** ** ** ** ** ** ** ** ** ** ** **


    # VALIDATE THRESHOLDS & PARAM_GRID ** ** ** ** ** ** ** ** ** **
    def threshold_checker(_thresholds, is_from_kwargs, idx):
        base_msg = (f"must be (1 - a list-type of 1 or more numbers) "
                    f"or (2 - a single number) and 0 <= number(s) <= 1")
        if is_from_kwargs:
            _msg = f"thresholds passed as a kwarg " + base_msg
        else:
            _msg = f"thresholds passed as param to param_grid[{idx}] " + base_msg
        del base_msg

        if _thresholds is None:
            _thresholds = np.linspace(0, 1, 21)
        else:
            try:
                _thresholds = np.array(list(_thresholds), dtype=np.float64)
            except:
                try:
                    int(_thresholds);
                    _thresholds = np.array([_thresholds], dtype=np.float64)
                except:
                    _exc(_msg)

        if len(_thresholds) == 0:
            _exc(_msg)

        for _thresh in _thresholds:
            try:
                int(_thresh)
            except:
                _exc(_msg)

            if not (_thresh >= 0 and _thresh <= 1):
                _exc(_msg)

        _thresholds.sort()

        del _msg
        return _thresholds


    _msg = (f"param_grid must be a (1 - dictionary) or (2 - a list of "
            f"dictionaries) with strings as keys and lists as values")

    if self.param_grid is None:
        self.wip_param_grid = [{}]
    elif isinstance(self.param_grid, dict):
        self.wip_param_grid = [self.param_grid]
    else:
        try:
            self.wip_param_grid = list(self.param_grid)
        except:
            _exc(_msg)

    # param_grid must be list at this point
    for grid_idx, _grid in enumerate(self.wip_param_grid):
        if not isinstance(_grid, dict):
            _exc(_msg)
        for k, v in _grid.items():
            if not 'str' in str(type(k)).lower():
                _exc(_msg)
            try:
                v = compute(v, scheduler=self.scheduler)[0]
            except:
                pass
            try:
                v = list(v)
            except:
                _exc(_msg)
            self.wip_param_grid[grid_idx][k] = np.array(v)

    del _msg

    # at this point param_grid must be a list of dictionaries with
    # str as keys and eager lists as values
    for grid_idx, _grid in enumerate(self.wip_param_grid):

        if np.fromiter(map(lambda x: 'threshold' in x,
                           list(map(str.lower, _grid.keys()))),
                       dtype=bool).sum() > 1:
            _exc(
                f"there are multiple keys in param_dict[{grid_idx}] indicating threshold")

        new_grid = {}
        for _key, _value in _grid.items():
            if 'threshold' in _key.lower():
                new_grid['thresholds'] = _value
            else:
                new_grid[_key] = _value

        _grid = new_grid
        del new_grid

        if 'thresholds' in _grid:
            _grid['thresholds'] = threshold_checker(_grid['thresholds'], False,
                                                    grid_idx)

        elif 'thresholds' not in _grid:
            _grid['thresholds'] = threshold_checker(self.thresholds, True, None)

        self.wip_param_grid[grid_idx] = _grid

    del threshold_checker, _grid
    # END VALIDATE THRESHOLDS & PARAM_GRID ** ** ** ** ** ** ** ** *

    # VALIDATE scoring / BUILD self.scorer_ ** ** ** ** ** ** ** **
    """
    scoring:
    Strategy to evaluate the performance of the cross-validated model on the tests set.
    If scoring represents a single score, one can use:
        a single string;
        a callable with signature (y_true, y_pred) that returns a single value.
    If scoring represents multiple scores, one can use:
        a list-type of unique strings;
        a list-type of (callable with signature (y_true, y_pred) that returns a single value);
        a dictionary with metric names as keys and callable(y_true, y_pred) as values;
        a callable returning (a dictionary with metric names as keys and callable(y_true, y_pred) as values)
    """

    ALLOWED_SCORING_DICT = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'average_precision': average_precision_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }


    def string_validation(_string: str):
        _string = _string.lower()
        if _string not in ALLOWED_SCORING_DICT:
            if 'roc_auc' in _string or 'average_precision' in _string:
                _exc(f"Dont need to use GridSearchThreshold when "
                     f"scoring is roc_auc or average_precision (auc_pr). "
                     f"Use regular dask/sklearn GridSearch and use max(tpr-fpr) "
                     f"to find best threshold for roc_auc, "
                     f"or use max(f1) to find the best threshold for "
                     f"average_precision.")
            else:
                raise NotImplementedError(
                    f"When specifying scoring by scorer name, must be "
                    f"in {', '.join(list(ALLOWED_SCORING_DICT))} ('{_string}')")

        return _string


    def check_callable_is_valid_metric(fxn_name: str, _callable: callable):
        _truth = np.random.randint(0, 2, (100,))
        _pred = np.random.randint(0, 2, (100,))
        try:
            _value = _callable(_truth, _pred)
        except:
            _exc(f"scoring function '{fxn_name}' excepted during validation")
        if not (_value >= 0 and _value <= 1):
            _exc(f"metric scoring function must have (y_true, y_pred) "
                 f"signature and return a number 0 <= number <= 1")
        del _value


    _msg = (f"scoring must be "
            f"\n1) a single string, or "
            f"\n2) a callable(y_true, y_pred) that returns a single value and 0 <= value <= 1, or "
            f"\n3) a list-type of strings, or "
            f"\n4) a dict of: (metric name: callable(y_true, y_pred), ...)."
            f"\nCannot pass None or bool. Cannot use estimator's default scorer.")

    _scoring = self.scoring

    try:
        _scoring = compute(_scoring, scheduler=self.scheduler)[0]
    except:
        pass

    if isinstance(_scoring, str):
        _scoring = string_validation(_scoring)
        _scoring = {_scoring: ALLOWED_SCORING_DICT[_scoring]}

    elif isinstance(_scoring, (list, tuple, set, np.ndarray)):
        try:
            _scoring = np.array(_scoring)
        except:
            _exc(_msg)
        _scoring = list(_scoring.ravel())
        if len(_scoring) == 0:
            _exc(f'scoring is empty --- ' + _msg)
        for idx, string_thing in enumerate(_scoring):
            if not isinstance(string_thing, str):
                _exc(_msg)
            _scoring[idx] = string_validation(string_thing)

        _scoring = list(set(_scoring))

        _scoring = {k: v for k, v in ALLOWED_SCORING_DICT.items() if k in _scoring}

    elif isinstance(_scoring, dict):
        if len(_scoring) == 0:
            _exc(f'scoring is empty --- ' + _msg)
        for key in _scoring:
            # DONT USE string_validation() HERE, USER-DEFINED CALLABLES CAN HAVE USER-DEFINED NAMES
            new_key = key.lower()
            _scoring[new_key] = _scoring.pop(key)
            check_callable_is_valid_metric(new_key, _scoring[new_key])
        del new_key

    elif callable(_scoring):
        check_callable_is_valid_metric(f'score', _scoring)
        _scoring = {f'score': _scoring}

    else:
        _exc(_msg)

    del _msg
    del string_validation, check_callable_is_valid_metric

    """
    dict of functions - Scorer function used on the held out data to 
    choose the best parameters for the model.
    A dictionary of {scorer_name: scorer} when one or multiple metrics 
    are used.
    """

    self.scorer_ = _scoring
    del _scoring

    self.multimetric_ = False if len(self.scorer_) == 1 else True

    del ALLOWED_SCORING_DICT
    # END VALIDATE scoring / BUILD self.scorer_ ** ** ** ** ** ** **



    # VALIDATE n_jobs ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    _val_n_jobs(self.n_jobs)
    # END VALIDATE n_jobs ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # VALIDATE pre_dispatch ** ** ** ** ** ** ** ** ** ** ** ** ** *
    if __is_dask_estimator
        try:
            del self.pre_dispatch
        except:
            pass
    else:
        # PIZZA FIGURE THIS OUT AND FIX IT
        pass
    # END VALIDATE pre_dispatch ** ** ** ** ** ** ** ** ** ** ** **

    # VALIDATE cv (n_splits_) ** ** ** ** ** ** ** ** ** ** ** ** **
    # UPHOLD THE dask/sklearn PRECEDENT THAT n_splits_ IS NOT AVAILABLE
    # AFTER init(), BUT IS AFTER fit()
    self.n_splits_ = self.cv or 5
    if not self.n_splits_ in range(2, 101): _exc(
        f"cv must be an integer in range(2,101)")
    # END VALIDATE cv ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

    # VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    self.wip_refit = self.refit

    try:
        self.wip_refit = compute(self.wip_refit, scheduler=self.scheduler)[0]
    except:
        pass

    _msg = (f"refit must be \n1) bool, \n2) None, \n3) a scoring "
            f"metric name, \n4) a callable that returns an integer, or "
            f"\n5) a list-type of length 1 that contains one of those "
            f"four."
            )

    if isinstance(self.wip_refit, dict):
        _exc(_msg)


    def type_getter(refit_arg, _return):
        _is_bool = isinstance(refit_arg, bool)
        _is_str = isinstance(refit_arg, str)
        _is_none = refit_arg is None
        _is_callable = callable(refit_arg)
        if not _return:
            return (_is_bool or _is_str or _is_none or _is_callable)
        return _is_bool, _is_str, _is_none, _is_callable


    if type_getter(self.wip_refit, False):
        _is_bool, _is_str, _is_none, _is_callable = type_getter(self.wip_refit,
                                                                True)
    else:
        try:
            self.wip_refit = list(self.wip_refit)
            if len(self.wip_refit) > 1:
                _exc(_msg)
            else:
                self.wip_refit = self.wip_refit[0]
            if type_getter(self.wip_refit, False):
                _is_bool, _is_str, _is_none, _is_callable = type_getter(
                    self.wip_refit, True)
            else:
                _exc(_msg)
        except:
            _exc(_msg)

    del type_getter

    if self.wip_refit is None:
        self.wip_refit = False
        _is_bool = True
        del _is_none

    if self.wip_refit is False or _is_callable:
        # CANT VALIDATE CALLABLE OUTPUT HERE, cv_results_ IS NOT AVAILABLE

        # AS OF 24_02_25_12_22_00 THERE ISNT A WAY TO RETURN A best_threshold_ WHEN MULTIPLE SCORERS ARE PASSED
        # TO scoring AND refit IS False OR A CALLABLE (OK IF REFIT IS A STRING).  IF THIS EVER CHANGES, THIS
        # WARNING CAN COME OUT.
        if len(self.scorer_) > 1:
            warnings.warn(
                f"WHEN MULTIPLE SCORERS ARE USED:\n"
                f"Cannot return a threshold if refit is False or callable.\n"
                f"If refit is False: best_index_, best_estimator_, best_score_, and best_threshold_ are not available.\n"
                f"if refit is callable: best_score_ and best_threshold_ are not available.\n"
                f"In either case, access score and threshold info via the cv_results_ attribute."
            )

        del _is_bool, _is_str, _is_callable
        pass
    else:  # refit CAN BE True OR (MATCHING A STRING IN scoring) ONLY
        refit_is_true = self.wip_refit == True
        refit_is_str = _is_str
        del _is_bool, _is_str, _is_callable

        _msg = lambda \
            x: f"egregious coding failure - refit_is_str and refit_is_bool are both {x}"
        if refit_is_str and refit_is_true:
            _exc(_msg(True))
        elif not refit_is_str and not refit_is_true:
            _exc(_msg(False))
        del _msg

        if refit_is_str:
            self.wip_refit = self.wip_refit.lower()

        # self.scorer_ KEYS CAN ONLY BE SINGLE STRINGS: user-defined via dict, 'score', or actual score method name
        if refit_is_true:
            if len(self.scorer_) == 1:
                self.wip_refit = 'score'
            elif len(self.scorer_) > 1:
                _exc(f"when scoring is multiple metrics, refit must be: "
                     f"\n1) a single string exactly matching a scoring method in scoring, "
                     f"\n2) a callable that returns an index in cv_results_ to use as best_index_, "
                     f"\n3) False")
        elif refit_is_str:
            if self.wip_refit not in self.scorer_:
                if len(self.scorer_) == 1:
                    _exc(
                        f"if using a single scoring metric, the allowed entries for refit are True, False, a callable "
                        f"that returns a best_index_, or a string that exactly matches the string passed to scoring")
                elif len(self.scorer_) > 1:
                    _exc(
                        f"if refit is a string, refit must exactly match one of the scoring methods in scoring")
            elif len(self.scorer_) == 1:
                self.wip_refit = 'score'

        del refit_is_true, refit_is_str

    # IF AN INSTANCE HAS ALREADY BEEN fit() WITH refit != False,
    # POST-REFIT ATTRS WILL BE AVAILABLE. BUT IF SUBSEQUENTLY refit
    # IS SET TO FALSE VIA set_params, THE POST-REFIT ATTRS NEED TO BE
    # REMOVED. IN THIS CONFIGURATION (i) THE REMOVE WILL HAPPEN AFTER
    # fit() IS CALLED AGAIN WHERE THIS METHOD (val&reset) IS CALLED
    # NEAR THE TOP OF fit() (ii) SETTING NEW PARAMS VIA set_params()
    # WILL LEAVE POST-REFIT ATTRS AVAILABLE ON AN INSTANCE THAT SHOWS
    # CHANGED PARAM SETTINGS (AS VIEWED VIA get_params()) UNTIL fit()
    # IS RUN.
    if self.wip_refit is False:
        try:
            del self.best_estimator_
        except:
            pass
        try:
            del self.refit_time_
        except:
            pass
        try:
            del self.feature_names_in_
        except:
            pass

    # END VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

    # NOW THAT refit IS VALIDATED, IF ONE THING IN SCORING, CHANGE THE KEY TO 'score'
    if len(self.scorer_) == 1:
        self.scorer_ = {'score': v for k, v in self.scorer_.items()}

    # VALIDATE verbose ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # pizza, y "if not __is_dask_estimator"????
    if not __is_dask_estimator:
        self.verbose = _val_verbose(self.verbose)
    # END VALIDATE verbose ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # VALIDATE error_score ** ** ** ** ** ** ** ** ** ** ** ** ** **
    if isinstance(self.error_score, str):
        self.error_score = self.error_score.lower()
        if not self.error_score == 'raise':
            _exc(
                f"the only string that can be passed to kwarg error_score is 'raise'")
    else:
        _msg = f"kwarg error_score must be 1) 'raise', 2) a number 0 <= number <= 1, 3) np.nan"
        if self.error_score is np.nan:
            pass
        else:
            try:
                np.float64(self.error_score)
            except:
                _exc(_msg)
            if not (self.error_score >= 0 and self.error_score <= 1):
                _exc(_msg)
        del _msg
    # END VALIDATE error_score ** ** ** ** ** ** ** ** ** ** ** ** *

    # VALIDATE return_train_score ** ** ** ** ** ** ** ** ** ** ** *
    if self.return_train_score is None:
        self.return_train_score = False
    if not isinstance(self.return_train_score, bool):
        _exc(f"return_train_score must be True, False, or None")
    # END VALIDATE return_train_score ** ** ** ** ** ** ** ** ** **

    # OTHER POSSIBLE KWARGS FOR DASK SUPPORT
    # VALIDATE iid ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    if __is_dask_estimator:
        if not isinstance(self.iid, bool):
            _exc(f'kwarg iid must be a bool')
    else:
        try:
            del self.iid
        except:
            pass
    # END VALIDATE iid ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # VALIDATE scheduler ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    if __is_dask_estimator:
        if self.scheduler is None:
            # If no special scheduler is passed, use a n_jobs local cluster
            self.scheduler = Client(n_workers=self.n_jobs, threads_per_worker=1,
                                    set_as_default=True)
        else:
            # self.scheduler ONLY FLOWS THRU TO compute(), SO LET compute() HANDLE VALIDATION
            pass
    else:
        try:
            del self.scheduler
        except:
            pass
    # END VALIDATE scheduler ** ** ** ** ** ** ** ** ** ** ** ** **

    # VALIDATE cache_cv ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    if __is_dask_estimator:
        if not isinstance(self.cache_cv, bool):
            _exc(f'kwarg cache_cv must be a bool')
    else:
        try:
            del self.cache_cv
        except:
            pass
    # END VALIDATE cache_cv ** ** ** ** ** ** ** ** ** ** ** ** ** *

    self.fit_excepted = False

    del _exc

    # END validate_and_reset ###########################################




































