# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from copy import deepcopy
import time
import warnings
from typing import Union

import numpy as np
import dask.array as da

from ._validation._cv import _validate_cv
from ._validation._error_score import _validate_error_score
from ._validation._verbose import _validate_verbose
from ._validation._n_jobs import _validate_n_jobs
from ._validation._refit import _validate_refit
from ._validation._return_train_score import _validate_return_train_score
from ._validation._scoring import _validate_scoring
from ._validation._thresholds__param_grid import _validate_thresholds__param_grid

class _GSTCVMixin:

    """

    --- Classifer must have predict_proba method. If does not have predict_proba,
    try to wrap with CalibratedClassifierCV.



    Notes
    -----
    The parameters selected are those that maximize the score of the left
    out data, unless an explicit score is passed in which case it is
    used instead.  ???


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



    # END init ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # pizza 24_07_26_08_43_00 what to do about this? it can probably go
    # @property
    # def estimator(self):
    #     return self._estimator

    # pizza 24_07_26_08_39_00 if the get/set_params stuff works out with
    # self.estimator not being self._estimator, then this can come out
    # @estimator.setter
    # def estimator(self, value):
    #     self.set_estimator_error()


    # def __repr__(self, N_CHAR_MAX=700):
    #
    #     # PIZZA, ALL OF __repr__ IS COPIED AND PASTED DIRECTLY FROM sklearn
    #     # BaseEstimator 24_07_28_14_21_00
    #
    #
    #     # N_CHAR_MAX is the (approximate) maximum number of non-blank
    #     # characters to render. We pass it as an optional parameter to ease
    #     # the tests.
    #     import re
    #     from sklearn.utils._pprint import _EstimatorPrettyPrinter
    #
    #     N_MAX_ELEMENTS_TO_SHOW = 30  # number of elements to show in sequences
    #
    #     # use ellipsis for sequences with a lot of elements
    #     pp = _EstimatorPrettyPrinter(
    #         compact=True,
    #         indent=1,
    #         indent_at_name=True,
    #         n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW,
    #     )
    #
    #     repr_ = pp.pformat(self)
    #
    #     # Use bruteforce ellipsis when there are a lot of non-blank characters
    #     n_nonblank = len("".join(repr_.split()))
    #     if n_nonblank > N_CHAR_MAX:
    #         lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
    #         regex = r"^(\s*\S){%d}" % lim
    #         # The regex '^(\s*\S){%d}' % n
    #         # matches from the start of the string until the nth non-blank
    #         # character:
    #         # - ^ matches the start of string
    #         # - (pattern){n} matches n repetitions of pattern
    #         # - \s*\S matches a non-blank char following zero or more blanks
    #         left_lim = re.match(regex, repr_).end()
    #         right_lim = re.match(regex, repr_[::-1]).end()
    #
    #         if "\n" in repr_[left_lim:-right_lim]:
    #             # The left side and right side aren't on the same line.
    #             # To avoid weird cuts, e.g.:
    #             # categoric...ore',
    #             # we need to start the right side with an appropriate newline
    #             # character so that it renders properly as:
    #             # categoric...
    #             # handle_unknown='ignore',
    #             # so we add [^\n]*\n which matches until the next \n
    #             regex += r"[^\n]*\n"
    #             right_lim = re.match(regex, repr_[::-1]).end()
    #
    #         ellipsis = "..."
    #         if left_lim + len(ellipsis) < len(repr_) - right_lim:
    #             # Only add ellipsis if it results in a shorter repr
    #             repr_ = repr_[:left_lim] + "..." + repr_[-right_lim:]
    #
    #     return repr_




    @property
    def classes_(self):
        """
        classes_: ndarray of shape (n_classes,) - Class labels. Only
            available when refit is not False and the estimator is a
            classifier (must be classifier --- validated in
            _validate_and_reset())


        """
        self.check_refit_is_false_no_attr_no_method('classes_')

        # pizza put this here 24_07_28_17_00_00
        # self.check_is_fitted()

        if not hasattr(self, '_classes'):   # MEANS fit() HASNT BEEN RUN
            self.estimator_hasattr('classes_')   # GET THE EXCEPTION

        if self._classes is None:
            # self._classes IS A HOLDER THAT IS FILLED ONE TIME WHEN
            # THIS METHOD IS CALLED
            # self._classes_ IS y AND CAN ONLY BE np OR da ARRAY
            if hasattr(self._estimator, 'classes_'):
                self._classes = self.best_estimator_.classes_
            else:
                # 24_02_25_16_13_00 --- GIVES USER ACCESS TO THIS ATTR FOR DASK
                # da.unique WORKS ON np AND dask arrays
                self._classes = da.unique(self._classes_).compute()

            del self._classes_

        return self._classes


    @property
    def n_features_in_(self):
        """
            property n_features_in_: Number of features seen during fit.
            Only available when refit is not False.

        """

        try:
            self.check_is_fitted()
        except:
            __ = type(self).__name__
            raise Exception(f"{__} object has no n_features_in_ attribute.")

        if self._n_features_in is None:
            # self._n_features_in IS A HOLDER THAT IS FILLED ONE TIME
            # WHEN THIS METHOD IS CALLED
            # self._n_features_in_ IS X.shape[0] AND CAN BE int OR delayed
            if hasattr(self._estimator, 'n_features_in_'):
                self._n_features_in = self.best_estimator_.n_features_in_
            elif self.wip_refit is not False:
                # 24_02_25_17_07_00 --- GIVES USER ACCESS TO THIS ATTR FOR DASK
                try: self._n_features_in = self._n_features_in_.compute()
                except: self._n_features_in = self._n_features_in_

            del self._n_features_in_

        return self._n_features_in


    ####################################################################
    # SKLEARN / DASK GridSearchCV Methods ##############################

    def fit(self, _X, _y, **refit_params):

        # post_fit_supplemental

        """
        refit:
            if False:
            self.best_index_: Indeterminate, not explicitly stated. Most likely is available.
            self.best_estimator_: estimator --- Not available if refit=False.
            self.best_score_: is available as a float
            self.best_params_: For multi-metric evaluation, not present if refit is not specified.
            self.n_features_in_: Not available.

            if not False:

                expose refit_time_ attr --- Seconds used for refitting the best model on the whole dataset.



                the refitted estimator is made available at the best_estimator_ attribute and
                                permits using predict directly on this GridSearchCV instance.

                one metric:
                    True --- maximum score is used to choose the best estimator
                             best_index_ is available
                             best_estimator is available, set according to the returned best_index_
                             best_score_ is available as a float
                             best_params_ is available, set according to the returned best_index_
                             self.n_features_in_: sklearn only. Only available when refit = True (single metric)

                multi metric:
                    str --- maximum score for that metric is used to choose the best estimator
                            best_index_ is available
                            best_estimator is available, set according to the returned best_index_
                            best_score_ is available as a float!  THE DASK DOCS ARE INCORRECT!  NO DICTIONARY EVER!
                            best_params_ is available, set according to the returned best_index_

                both:
                    a function which returns the selected best_index_ when passed cv_results_
                        ---best_index is available
                        ---best_estimator_ is available, set according to the returned best_index_
                        ---best_score_ attribute will not be available.
                        ---best_params_ is available, set according to the returned best_index_
        """

        """
        self.best_index_: int or dict of ints
        The index of the cv_results_ array which corresponds to the best candidate parameter setting.
        This locates the dict in search.cv_results_['params'][search.best_index_] holding the parameter settings
        for the best model, i.e., the one that gives the highest mean score (search.best_score_) on the holdout data.
        When using multiple metrics, best_index_ will be a dictionary where the keys are the names of the scorers,
        and the values are the index with the best mean score for that scorer, as described above.
        """

        """
        self.best_estimator_: estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss
        if specified) on the left out data. Not available if refit=False.
        """

        """
        self.best_score_: float
            THE DASK DOCS ARE INCORRECT!  NO DICTIONARY EVER!
            Mean tests score of best_estimator on the hold out data.
        """

        """
        self.best_params_: dict
            The dict at search.cv_results_['params'][search.best_index_] that holds the parameter settings that yields
            the best model (i.e, gives the highest mean score -- search.best_score_ -- on the hold out data.)
            For multi-metric evaluation, this is present only if refit is specified.
        """

        """
        self.n_features_in_:
            sklearn only
            Number of features seen during fit. Only available when refit = True.
        """

        """
        self.refit_time_
            Seconds used for refitting the best model on the whole dataset.
            This is present only if refit is not False.
        """

        refit_is_str = isinstance(self.wip_refit, str)
        refit_is_false = self.wip_refit == False
        refit_is_callable = callable(self.wip_refit)

        if refit_is_callable:
            refit_fxn_output = self.wip_refit(deepcopy(self.cv_results_))
            _msg = (f"if a callable is passed to refit, it must yield or return "
                    f"an integer, and it must be within range of cv_results_ rows.")
            if not int(refit_fxn_output) == refit_fxn_output:
                raise ValueError(_msg)
            if refit_fxn_output > self.cv_results_['params'].shape[0]:
                raise ValueError(_msg)
            self.best_index_ = refit_fxn_output
            del refit_fxn_output
            self.best_params_ = self.cv_results_['params'][self.best_index_]
            if len(self.scorer_) == 1:
                self.best_threshold_ = self.cv_results_['best_threshold'][self.best_index_]
                self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]
            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass
        elif refit_is_false:
            if len(self.scorer_) == 1:
                self.best_index_ = np.arange(self.cv_results_['params'].shape[0])[self.cv_results_['rank_test_score'] == 1][0]
                self.best_params_ = self.cv_results_['params'][self.best_index_]
                self.best_threshold_ = self.cv_results_['best_threshold'][self.best_index_]
                self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]
                # pizza verified best_score_ 24_07_16 in jupyter through various experiments
                # really is mean_test_score for best_index.
            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass
        elif refit_is_str:
            # DOESNT MATTER WHAT len(self.scorer_) IS
            self.best_index_ = np.arange(self.cv_results_['params'].shape[0])[self.cv_results_[f'rank_test_{self.wip_refit}'] == 1][0]
            self.best_params_ = self.cv_results_['params'][self.best_index_]
            threshold_column = f'best_threshold' if len(self.scorer_)==1 else f'best_threshold_{self.wip_refit}'
            self.best_threshold_ = self.cv_results_[threshold_column][self.best_index_]
            del threshold_column
            self.best_score_ = self.cv_results_[f'mean_test_{self.wip_refit}'][self.best_index_]

        del refit_is_str, refit_is_callable, refit_is_false

        if self.wip_refit:
            if self.verbose >= 3: print(f'\nStarting refit...')
            self.best_estimator_ = self._estimator
            self.best_estimator_.set_params(**self.best_params_)
            t0 = time.perf_counter()

            self.best_estimator_.fit(_X, _y, **refit_params)
            self.refit_time_ = time.perf_counter() - t0
            del t0
            if self.verbose >= 3:
                print(f'Finished refit. time = {self.refit_time_}')

            # feature_names_in_: ndarray of shape (n_features_in_,)
            # Names of features seen during fit. Only defined if
            # best_estimator_ is defined and if best_estimator_ exposes
            # feature_names_in_ when fit.
            try:
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
            except:
                try:
                    self.feature_names_in_ = self._feature_names_in
                except:
                    pass

        elif self.wip_refit is False:
            pass

        return self








    def decision_function(self, X):
        """
        Call decision_function on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports decision_function.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_score: ndarray of shape (n_samples,) or (n_samples, n_classes) or
            (n_samples, n_classes * (n_classes-1) / 2) - Result of the
            decision function for X based on the estimator with the best
            found parameters.


        """

        self.estimator_hasattr('predict_proba')

        self.check_refit_is_false_no_attr_no_method('predict_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.decision_function(X)


    def get_metadata_routing(self):
        """
        Get metadata routing of this object.
        Please check User Guide on how the routing mechanism works.

        Return
        ------
        -
            routingMetadataRouter: A MetadataRouter encapsulating routing
                information.
        """

        # sklearn only --- always available, before and after fit()

        __ = type(self).__name__
        raise NotImplementedError(f"get_metadata_routing is not implemented in {__}.")


    def get_params(self, deep:bool=True):

        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Return
        ------
        -
            paramsdict: Parameter names mapped to their values.

        """

        # sklearn / dask -- this is always available, before & after fit

        if not isinstance(deep, bool):
            raise ValueError(f"'deep' must be boolean")

        paramsdict = deepcopy(vars(self))

        # gymnastics to get GSTCV param order the same as sk/dask GSCV
        paramsdict1 = {}
        paramsdict2 = {}
        key = 0
        for k in sorted(paramsdict):
            if k == 'estimator':
                key = 1
            if key == 0:
                paramsdict1[k] = paramsdict.pop(k)
            else:
                paramsdict2[k] = paramsdict.pop(k)
        del key


        if deep:
            estimator_params = {}
            for k, v in deepcopy(paramsdict2['estimator'].get_params()).items():
                estimator_params[f'estimator__{k}'] = v

            paramsdict1 = paramsdict1 | estimator_params


        paramsdict = paramsdict1 | paramsdict2

        del paramsdict1, paramsdict2

        return paramsdict


    def inverse_transform(self, Xt):
        """
        Call inverse_transform on the estimator with the best found params.
        Only available if the underlying estimator implements
        inverse_transform and refit is not False.

        Parameters
        ----------
        Xt: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        -
            X Union[ndarray, sparse matrix, pizza] of shape (n_samples,
                n_features) - Result of the inverse_transform function
                for Xt based on the estimator with the best found parameters.


        """

        # PIZZA 24_07_01_08_44_00 FIGURE OUT IF THIS SHOULD STAY OR GO

        self.estimator_hasattr('inverse_transform')

        self.check_refit_is_false_no_attr_no_method('inverse_transform')

        return self.best_estimator_.inverse_transform(Xt)


    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Parameters
        ----------
        X: pizza

        Return
        ------
        pizza
            The predicted labels or values for X based on the estimator
            with the best found parameters.

        """

        self.estimator_hasattr('predict')

        self.check_refit_is_false_no_attr_no_method('predict')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        y_pred = self.best_estimator_.predict_proba(X)[:, -1] > self.best_threshold_

        return y_pred.astype(np.uint8)


    def predict_log_proba(self, X):
        """
        Call predict_log_proba on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports predict_log_proba.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_pred: ndarray of shape (n_samples,) or (n_samples, n_classes) -
            Predicted class log-probabilities for X based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute classes_.
        """

        self.estimator_hasattr('predict_log_proba')

        self.check_refit_is_false_no_attr_no_method('predict_log_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.predict_log_proba(X)


    def predict_proba(self, X):
        """
        Call predict_proba on the estimator with the best found parameters.
        Only available if refit is not False and the underlying estimator
        supports predict_proba.

        Parameters
        ----------
        X: indexable, length n_samples - Must fulfill the input
            assumptions of the underlying estimator.

        Return
        ------
        y_pred: ndarray of shape (n_samples,) or (n_samples, n_classes) -
            Predicted class probabilities for X based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute classes_.


        """

        self.estimator_hasattr('predict_proba')

        self.check_refit_is_false_no_attr_no_method('predict_proba')

        self.check_is_fitted()

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.predict_proba(X)


    def score(self, X, y=None, **params):
        """
        Return the score on the given data, if the estimator has been
        refit. This uses the score defined by scoring where provided, and
        the best_estimator_.score method otherwise.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) - Input data,
            where n_samples is the number of samples and n_features is
            the number of features.
        y: array-like of shape (n_samples, n_output) or (n_samples,),
            default=None - Target relative to X for classification or
            regression; None for unsupervised learning.
        **params: dict of parameters to be passed to the underlying
            scorer(s). Only available if enable_metadata_routing=True.
            See Metadata Routing User Guide for more details.

        Return
        ------
        score: float - The score defined by scoring if provided, and the
            best_estimator_.score method otherwise.
        """

        self.estimator_hasattr('score')

        self.check_refit_is_false_no_attr_no_method('score')

        self.check_is_fitted()

        X, y = self._handle_X_y(X, y=y)[[0,1]]

        y_pred = (self.predict_proba(X)[:, -1] >= self.best_threshold_)

        return self.scorer_[self.wip_refit](y, y_pred, **params)


    def score_samples(self, X):
        """Call score_samples on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports score_samples. New in version 0.24.

        Parameters
        ----------
        X: iterable - Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Return
        ------
        -
            The best_estimator_.score_samples method.

        """

        self.estimator_hasattr('score_samples')

        self.check_refit_is_false_no_attr_no_method('score_samples')

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.score_samples(X)


    def set_params(self, **params):
        
        """
        Set the parameters of this estimator. The method works on simple
        estimators as well as on nested objects (such as Pipeline). The
        latter have parameters of the form <component>__<parameter> so
        that it’s possible to update each component of a nested object.

        Parameters
        ----------
        **params: dict[str: any] - Estimator parameters.

        Return
        ------
        -
            GridSearchThresholdCV (GSTCV) instance.

        """

        # estimators, pipelines, and gscv all raise exception for invalid
        # keys (parameters) passed

        # make lists of what parameters are valid
        # use shallow get_params to get valid params for GSTCV
        ALLOWED_GSTCV_PARAMS = self.get_params(deep=False)
        # use deep get_params to get valid params for estimator/pipe
        ALLOWED_EST_PARAMS = {}
        for k, v in self.get_params(deep=True).items():
            if k not in ALLOWED_GSTCV_PARAMS:
                ALLOWED_EST_PARAMS[k.replace('estimator__', '')] = v


        # separate estimator and GSTCV parameters
        est_params = {}
        gstcv_params = {}
        for k,v in params.items():
            if 'estimator__' in k:
                est_params[k.replace('estimator__', '')] = v
            else:
                gstcv_params[k] = v
        # END separate estimator and GSTCV parameters


        def _invalid_est_param(parameter: str, ALLOWED: dict) -> None:
            raise ValueError(
                f"invalid parameter '{parameter}' for estimator "
                f"{type(self).__name__}(pipeline={ALLOWED['estimator']}, "
                f"param_grid={ALLOWED['param_grid']}). \n"
                f"Valid parameters are: {list(ALLOWED.keys())}"
            )


        # set GSTCV params
        # GSTCV(Dask) parameters must be validated & set the long way
        for gstcv_param in gstcv_params:
            if gstcv_param not in ALLOWED_GSTCV_PARAMS:
                raise ValueError(
                    _invalid_est_param(gstcv_param, ALLOWED_GSTCV_PARAMS)
                )
            setattr(self, gstcv_param, params[gstcv_param])


        # set estimator params ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # IF self.estimator is dask/sklearn est/pipe, THIS SHOULD HANDLE
        # EXCEPTIONS FOR INVALID PASSED PARAMS
        self.estimator.set_params(**est_params)

        # this is stop-gap validation in case an estimator (of a makeshift
        # sort, perhaps) does not block setting invalid params.
        for est_param in est_params:
            if est_param not in ALLOWED_EST_PARAMS:
                raise ValueError(
                    _invalid_est_param(est_param, ALLOWED_EST_PARAMS)
                )
        # END set estimator params ** * ** * ** * ** * ** * ** * ** * **

        del ALLOWED_EST_PARAMS, ALLOWED_GSTCV_PARAMS, _invalid_est_param

        return self


    def transform(self, X):
        """
        Call transform on the estimator with the best found parameters.
        Only available if the underlying estimator supports transform and
        refit is not False.

        Parameters
        ----------
        X: indexable, length n_samples. Must fulfill the input assumptions
            of the underlying estimator.

        Return
        ------
        Xt: {ndarray, sparse matrix} of shape (n_samples, n_features) -
            X transformed in the new space based on the estimator with
            the best found parameters.


        """


        self.estimator_hasattr('transform')

        self.check_refit_is_false_no_attr_no_method('transform')

        X = self._handle_X_y(X, y=None)[0]

        return self.best_estimator_.transform(X)


    # END SKLEARN / DASK GridSearchCV Method ###########################
    ####################################################################


    ####################################################################
    # SUPPORT METHODS ##################################################




    def _validate_and_reset(self):

        self.wip_param_grid = _validate_thresholds__param_grid(
            self.thresholds, self.param_grid
        )

        self.scorer_ = _validate_scoring(self.scoring)
        self.multimetric_ = len(self.scorer_) > 1

        # VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        self.wip_refit = _validate_refit(self.refit, self.scorer_)

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
                del  self.refit_time_
            except:
                pass
            try:
                del self.feature_names_in_
            except:
                pass

        # END VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # NOW THAT refit IS VALIDATED, IF ONE THING IN SCORING, CHANGE
        # THE KEY TO 'score'
        if len(self.scorer_)==1:
            self.scorer_ = {'score':v for k,v in self.scorer_.items()}

        # n_splits_ is only available after fit(). n_splits_ is always
        # returned as a number
        self._cv = _validate_cv(self.cv)
        try:
            float(self._cv)
            self.n_splits_ = self._cv
        except:
            self.n_splits_ = len(self._cv)

        self._error_score = _validate_error_score(self.error_score)

        self._verbose = _validate_verbose(self.verbose)

        self._return_train_score = \
            _validate_return_train_score(self.return_train_score)

        self._n_jobs = _validate_n_jobs(self.n_jobs)

    # END validate_and_reset ###########################################


    def estimator_hasattr(self, attr_or_method_name):
        if not hasattr(self._estimator, attr_or_method_name):
            raise AttributeError(f"'{type(self.estimator).__name__}' "
                     f"object has no attribute '{attr_or_method_name}'")
        else:
            return True


    def check_refit_is_false_no_attr_no_method(self, attr_or_method_name):
        if not self.refit:
            raise AttributeError(f"This GridSearchCV instance was initialized "
                f"with `refit=False`. {attr_or_method_name} is available only "
                f"after refitting on the best parameters. You can refit an "
                f"estimator manually using the `best_params_` attribute")
        else:
            return True


    def check_is_fitted(self):

        if not hasattr(self, 'wip_refit'):
            class NotFittedError(Exception):
                pass
            raise NotFittedError(f"This GridSearchCV instance is not "
                f"fitted yet. Call 'fit' with appropriate arguments "
                f"before using this estimator.")
        else:
            return True

    # END SUPPORT METHODS ##############################################
    ####################################################################





















