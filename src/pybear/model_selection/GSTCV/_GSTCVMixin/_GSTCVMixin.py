# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable
from typing_extensions import Union
from copy import deepcopy
import time
from sklearn.base import BaseEstimator
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


from .._fit_shared._cv_results._cv_results_builder import \
    _cv_results_builder

from .._fit_shared._verify_refit_callable import \
    _verify_refit_callable



class _GSTCVMixin(BaseEstimator):

    # BaseEstimator is intended to only provide __repr__.

    @property
    def classes_(self):

        """
        classes_: ndarray of shape (n_classes,) - Class labels. Only
            available when refit is not False.

        """

        self.check_refit__if_false_block_attr('classes_')

        self.check_is_fitted()

        if self._classes is None:
            # self._classes IS A HOLDER THAT IS FILLED ONE TIME WHEN
            # THIS METHOD IS CALLED
            # self._y IS y AND CAN ONLY BE np OR da ARRAY
            try:
                self._classes = getattr(self.best_estimator_, 'classes_')
            except:
                with self._scheduler:
                    # da.unique WORKS ON np AND dask arrays
                    self._classes = da.unique(self._y).compute()
                    del self._y


        return self._classes


    @property
    def n_features_in_(self):

        """
        n_features_in_: Number of features seen during fit. Only
        available when refit is not False.

        """

        __ = type(self).__name__

        try:
            self.check_is_fitted()
        except:
            raise AttributeError(f"{__} object has no n_features_in_ attribute.")

        # self._n_features_in_ IS X.shape[1] AND MUST BE int (DASK WAS COMPUTED)
        if self._refit is False:
            raise AttributeError(f"'{__}' object has no attribute 'n_features_in_'")
        else:
            if hasattr(self.best_estimator_, 'n_features_in_'):
                return self.best_estimator_.n_features_in_
            else:
                return self._n_features_in


    ####################################################################
    # SKLEARN / DASK GSTCV Methods #####################################

    def fit(
        self,
        X: Iterable[Iterable[Union[int, float]]],
        y: Iterable[int],
        **params
    ):

        """

        Perform the grid search with the hyperparameter settings in
        param grid to generate scores for the given X and y.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - The data on which to perform the grid search.
            Must contain all numerics. Must be able to convert to a
            numpy.ndarray (GSTCV) or dask.array.core.Array (GSTCVDask).
            Must fulfill the input assumptions of the underlying
            estimator.

        y:
            Iterable[int], shape (n_samples,) or (n_samples, 1) - The
            target relative to X. Must be binary in [0, 1]. Must be able
            to convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.

        **params:
            dict[str, any] - Parameters passed to the fit method of the
            estimator. If a fit parameter is an array-like whose length
            is equal to num_samples, then it will be split across CV
            groups along with X and y. For example, the sample_weight
            parameter is split because len(sample_weights) = len(X). For
            array-likes intended to be subject to CV splits, care must
            be taken to ensure that any such vector is shaped
            (num_samples, ) or (num_samples, 1), otherwise it will not
            be split. For GSTCVDask, pybear recommends passing such
            array-likes as dask arrays.

            For pipelines, fit parameters can be passed to the fit method
            of any of the steps. Prefix the parameter name with the name
            of the step, such that parameter p for step s has key s__p.


        Return
        ------
        -
            self: fitted estimator instance - GSTCV(Dask) instance.


        """


        self._validate_and_reset()

        # feature_names_in_: ndarray of shape (n_features_in_,)
        # Names of features seen during fit. Only defined if
        # best_estimator_ is defined and if best_estimator_ exposes
        # feature_names_in_ when fit.
        # try:
        _X, _y, _feature_names_in, self._n_features_in = self._handle_X_y(X, y)

        if _feature_names_in is not None and self._refit is not False:
            self.feature_names_in_ = _feature_names_in

        # DONT unique().compute() HERE, JUST RETAIN THE VECTOR & ONLY DO
        # THE PROCESSING IF classes_ IS CALLED
        self._y = _y.copy()

        # THIS IS A HOLDER THAT IS FILLED ONE TIME WHEN THE unique().compute()
        # IS DONE ON self._y WHEN @property classes_ IS CALLED
        self._classes = None


        # BEFORE RUNNING cv_results_builder, THE THRESHOLDS MUST BE
        # REMOVED FROM EACH PARAM GRID IN _param_grid BUT THEY NEED TO
        # BE RETAINED FOR CORE GRID SEARCH.
        # pop IS INTENTIONALLY USED HERE TO REMOVE 'thresholds' FROM
        # PARAM GRIDS.
        # 'thresholds' MUST BE REMOVED FROM PARAM GRIDS BEFORE GOING
        # TO _cv_results_builder OR THRESHOLDS WILL BECOME PART OF THE
        # GRID SEARCH, AND ALSO CANT BE PASSED TO estimator.
        self._THRESHOLD_DICT = {}
        for i in range(len(self._param_grid)):
            self._THRESHOLD_DICT[i] = self._param_grid[i].pop('thresholds')


        # this could have been at the top of _core_fit but is outside
        # because cv_results is used to validate the refit callable
        # before starting the fit.
        self.cv_results_, self._PARAM_GRID_KEY = \
            _cv_results_builder(
                self._param_grid,
                self.n_splits_,
                self.scorer_,
                self._return_train_score
        )

        # USE A DUMMIED-UP cv_results TO TEST IF THE refit CALLABLE RETURNS
        # A GOOD INDEX NUMBER, BEFORE RUNNING THE WHOLE GRIDSEARCH
        if callable(self._refit):
            _verify_refit_callable(self._refit, deepcopy(self.cv_results_))


        # CORE FIT v v v v v v v v v v v v v v v v v v v v v v v v v v v

        with self._scheduler as scheduler:
            self._core_fit(_X, _y, **params)

        # END CORE FIT ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        _rows = self.cv_results_['params'].shape[0]
        _get_best = lambda column: self.cv_results_[column][self.best_index_]
        _get_best_index = \
            lambda column: np.arange(_rows)[self.cv_results_[column] == 1][0]

        # 'refit' can be a str, bool False, or callable

        if callable(self._refit):

            refit_fxn_output = self._refit(deepcopy(self.cv_results_))
            _msg = (f"if a callable is passed to refit, it must yield or "
                    f"return an integer, and it must be within range of "
                    f"cv_results_ rows."
            )
            if not int(refit_fxn_output) == refit_fxn_output:
                raise ValueError(_msg)
            if refit_fxn_output > _rows:
                raise ValueError(_msg)


            self.best_index_ = int(refit_fxn_output)
            del refit_fxn_output

            self.best_params_ = _get_best('params')

            if len(self.scorer_) == 1:
                self.best_threshold_ = float(_get_best('best_threshold'))
                self.best_score_ = float(_get_best('mean_test_score'))

            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass

        elif self._refit == False:

            if len(self.scorer_) == 1:
                self.best_index_ = int(_get_best_index('rank_test_score'))
                self.best_params_ = _get_best('params')
                self.best_threshold_ = float(_get_best('best_threshold'))
                self.best_score_ = float(_get_best('mean_test_score'))
                # 24_07_16 through various experiments verified best_score_
                # really is mean_test_score for best_index
            elif len(self.scorer_) > 1:
                # A WARNING IS RAISED DURING VALIDATION
                # self.best_score_ NOT AVAILABLE
                # self.best_threshold_ NOT AVAILABLE
                pass

        elif isinstance(self._refit, str):
            # DOESNT MATTER WHAT len(self.scorer_) IS
            self.best_index_ = int(_get_best_index(f'rank_test_{self._refit}'))
            self.best_params_ = _get_best('params')

            if len(self.scorer_) == 1:
                threshold_column = f'best_threshold'
            else:
                threshold_column = f'best_threshold_{self._refit}'

            self.best_threshold_ = float(_get_best(threshold_column))
            del threshold_column
            self.best_score_ = float(_get_best(f'mean_test_{self._refit}'))
        else:
            raise Exception(f"invalid 'refit' value '{self._refit}'")

        del _rows
        del _get_best

        if self._refit:

            if self._verbose >= 3:
                print(f'\nStarting refit...')

            self.best_estimator_ = \
                type(self._estimator)(**self._estimator.get_params(deep=False))
            self.best_estimator_.set_params(**self._estimator.get_params(deep=True))

            del self._estimator

            self.best_estimator_.set_params(**self.best_params_)


            t0 = time.perf_counter()

            with self._scheduler as scheduler:
                self.best_estimator_.fit(_X, _y, **params)

            self.refit_time_ = time.perf_counter() - t0
            del t0
            if self._verbose >= 3:
                print(f'Finished refit. time = {self.refit_time_}')

        elif self._refit is False:
            pass


        return self


    def decision_function(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call decision_function on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports decision_function.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - Must contain all numerics. Must be able to
            convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.


        Return
        ------
        -
            The best_estimator_ decision_function method result for X.

        """

        self.estimator_hasattr('decision_function')

        self.check_refit__if_false_block_attr('decision_function')

        self.check_is_fitted()

        _X, passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:
            return self.best_estimator_.decision_function(_X)


    def get_metadata_routing(self):

        """

        get_metadata_routing is not implemented in GSTCV(Dask).

        """

        # sklearn only --- always available, before and after fit()

        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}."
        )


    # 24_10_27 now inheriting get_params from BaseEstimator
    # the below code previously worked in its stead, keep this for backup.
    # def get_params(self, deep:bool=True):
    #
    #     """
    #
    #     Get parameters for this GSTCV(Dask) instance.
    #
    #
    #     Parameters
    #     ----------
    #     deep:
    #         bool, optional, default=True - 'False' only returns the
    #         parameters of the GSTCV(Dask) instance. 'True' returns the
    #         parameters of the GSTCV(Dask) instance as well as the
    #         parameters of the estimator and anything embedded in the
    #         estimator. When the estimator is a single estimator, the
    #         parameters of the single estimator are returned. If the
    #         estimator is a pipeline, the parameters of the pipeline and
    #         the parameters of each of the steps in the pipeline are
    #         returned.
    #
    #
    #     Return
    #     ------
    #     -
    #         params: dict - Parameter names mapped to their values.
    #
    #     """
    #
    #     # sklearn / dask -- this is always available, before & after fit
    #
    #     if not isinstance(deep, bool):
    #         raise ValueError(f"'deep' must be boolean")
    #
    #     paramsdict = {}
    #     for attr in vars(self):
    #         # after fit, take out all the attrs with leading or trailing '_'
    #         if attr[0] == '_' or attr[-1] == '_':
    #             continue
    #
    #         if attr == 'scheduler': # cant pickle asyncio object
    #             paramsdict[attr] = self.scheduler
    #         else:
    #             paramsdict[attr] = deepcopy(vars(self)[attr])
    #
    #
    #     # gymnastics to get GSTCV param order the same as sk/dask GSCV
    #     paramsdict1 = {}
    #     paramsdict2 = {}
    #     key = 0
    #     for k in sorted(paramsdict):
    #         if k == 'estimator':
    #             key = 1
    #         if key == 0:
    #             paramsdict1[k] = paramsdict.pop(k)
    #         else:
    #             paramsdict2[k] = paramsdict.pop(k)
    #     del key
    #
    #
    #     if deep:
    #         estimator_params = {}
    #         for k, v in deepcopy(paramsdict2['estimator'].get_params()).items():
    #             estimator_params[f'estimator__{k}'] = v
    #
    #         paramsdict1 = paramsdict1 | estimator_params
    #
    #
    #     paramsdict = paramsdict1 | paramsdict2
    #
    #     del paramsdict1, paramsdict2
    #
    #     return paramsdict


    def inverse_transform(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call inverse_transform on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports inverse_transform.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]] - Must contain all
            numerics. Must be able to convert to a numpy.ndarray (GSTCV)
            or dask.array.core.Array (GSTCVDask). Must fulfill the input
            assumptions of the underlying estimator.


        Return
        ------
        -
            The best_estimator_ inverse_transform method result for X.


        """

        self.estimator_hasattr('inverse_transform')

        self.check_refit__if_false_block_attr('inverse_transform')

        self.check_is_fitted()

        with self._scheduler as scheduler:
            return self.best_estimator_.inverse_transform(X)


    def predict(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call the best estimator's predict_proba method on the passed X
        and apply the best_threshold_ to predict the classes for X. When
        only one scorer is used, predict is available if refit is not
        False. When more than one scorer is used, predict is only
        available if refit is set to a string.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - Must contain all numerics. Must be able to
            convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.


        Return
        ------
        -
            A vector in [0,1] indicating the class label for the examples
            in X. A numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask) is returned.

        """

        self.estimator_hasattr('predict')

        self.check_refit__if_false_block_attr('predict')

        self.check_is_fitted()

        if len(self.scorer_) > 1 and callable(self._refit):
            raise AttributeError(f"'predict' is not available when there "
                f"are multiple scorers and refit is a callable because "
                f"best_threshold_ cannot be determined.")

        _X, passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:

            y_pred = self.best_estimator_.predict_proba(_X)[:, -1] >= \
                        self.best_threshold_

            return y_pred.astype(np.uint8)


    def predict_log_proba(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call predict_log_proba on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports predict_log_proba.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - Must contain all numerics. Must be able to
            convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.


        Return
        ------
        -
            The best_estimator_ predict_log_proba method result for X.

        """

        self.estimator_hasattr('predict_log_proba')

        self.check_refit__if_false_block_attr('predict_log_proba')

        self.check_is_fitted()

        _X , passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:
            return self.best_estimator_.predict_log_proba(_X)


    def predict_proba(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call predict_proba on the estimator with the best found
        parameters. Only available if refit is not False. The underlying
        estimator must support this method, as it is a characteristic
        that is validated.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - Must contain all numerics. Must be able to
            convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.


        Return
        ------
        -
            The best_estimator_ predict_proba_ method result for X.

        """

        self.estimator_hasattr('predict_proba')

        self.check_refit__if_false_block_attr('predict_proba')

        self.check_is_fitted()

        _X, passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:

            __ = self.best_estimator_.predict_proba(_X)

            _shape = __.shape
            if len(_shape) != 2 or _shape[1] != 2:
                raise ValueError(
                    f"'predict_proba' output for X was expected to be 2 "
                    f"dimensional with 2 columns, but got shape {_shape} instead")
            del _shape

            return __


    def score(
        self,
        X: Iterable[Iterable[Union[int, float]]],
        y: Iterable[int]
    ):

        """

        Score the given X and y using the best estimator, best threshold,
        and the defined scorer. When there is only one scorer, that is
        the defined scorer, and score is available if refit is not False.
        When there are multiple scorers, the defined scorer is the scorer
        specified by 'refit', and score is available only if refit is set
        to a string.

        See the documentation for the 'scoring' parameter for information
        about passing kwargs to the scorer.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]], shape (n_samples,
            n_features) - Must contain all numerics. Must be able to
            convert to a numpy.ndarray (GSTCV) or dask.array.core.Array
            (GSTCVDask). Must fulfill the input assumptions of the
            underlying estimator.

        y:
            Iterable[Union[int, float]], shape (n_samples, ) or
            (n_samples, 1) - The target relative to X. Must be binary in
            [0, 1]. Must be able to convert to a numpy.ndarray (GSTCV)
            or dask.array.core.Array (GSTCVDask).


        Return
        ------
        -
            score:
                float - The score for X and y on the best estimator and
                best threshold using the defined scorer.

        """

        self.estimator_hasattr('score')

        self.check_refit__if_false_block_attr('score')

        self.check_is_fitted()

        if callable(self._refit) and len(self.scorer_) > 1:
            return self._refit

        _X, _y, passed_feature_names = self._handle_X_y(X, y=y)[:3]

        self._validate_features(passed_feature_names)

        y_pred = self.predict(_X)

        # if refit is False, score() is not would be accessible
        with self._scheduler as scheduler:

            if callable(self._refit) and len(self.scorer_) == 1:
                return self.scorer_['score'](_y, y_pred)
            # elif callable(self._refit) and len(self.scorer_) > 1:
            #   handled above
            else:
                return self.scorer_[self._refit](_y, y_pred)


    def score_samples(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call score_samples on the estimator with the best found
        parameters. Only available if refit is not False and the
        underlying estimator supports score_samples.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]] - Must contain all
            numerics. Must be able to convert to a numpy.ndarray (GSTCV)
            or dask.array.core.Array (GSTCVDask). Must fulfill the input
            assumptions of the underlying estimator.


        Return
        ------
        -
            The best_estimator_ score_samples method result for X.

        """

        self.estimator_hasattr('score_samples')

        self.check_refit__if_false_block_attr('score_samples')

        self.check_is_fitted()

        _X, passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:
            return self.best_estimator_.score_samples(_X)


    # 24_10_27 now inheriting set_params from BaseEstimator
    # the below code previously worked in its stead, keep this for backup.
    # def set_params(self, **params):
    #
    #     """
    #
    #     Set the parameters of the GSTCV(Dask) instance or the embedded
    #     estimator. The method works on simple estimators as well as on
    #     nested objects (such as Pipeline). The parameters of single
    #     estimators can be updated using 'estimator__<parameter>'.
    #     Pipeline parameters can be updated using the form
    #     'estimator__<pipe_parameter>. Steps of a pipeline have parameters
    #     of the form <step>__<parameter> so that it’s also possible to
    #     update a step's parameters. The parameters of steps in the
    #     pipeline can be updated using 'estimator__<step>__<parameter>'.
    #
    #
    #     Parameters
    #     ----------
    #     **params:
    #         dict[str: any] - GSTCV(Dask) and/or estimator parameters.
    #
    #
    #     Return
    #     ------
    #     -
    #         self: estimator instance - GSTCV(Dask) instance.
    #
    #     """
    #
    #     # estimators, pipelines, and gscv all raise exception for invalid
    #     # keys (parameters) passed
    #
    #     # make lists of what parameters are valid
    #     # use shallow get_params to get valid params for GSTCV
    #     ALLOWED_GSTCV_PARAMS = self.get_params(deep=False)
    #     # use deep get_params to get valid params for estimator/pipe
    #     ALLOWED_EST_PARAMS = {}
    #     for k, v in self.get_params(deep=True).items():
    #         if k not in ALLOWED_GSTCV_PARAMS:
    #             ALLOWED_EST_PARAMS[k.replace('estimator__', '')] = v
    #
    #
    #     # separate estimator and GSTCV parameters
    #     est_params = {}
    #     gstcv_params = {}
    #     for k,v in params.items():
    #         if 'estimator__' in k:
    #             est_params[k.replace('estimator__', '')] = v
    #         else:
    #             gstcv_params[k] = v
    #     # END separate estimator and GSTCV parameters
    #
    #
    #     def _invalid_est_param(parameter: str, ALLOWED: dict) -> None:
    #         raise ValueError(
    #             f"invalid parameter '{parameter}' for estimator "
    #             f"{type(self).__name__}(estimator={ALLOWED['estimator']}, "
    #             f"param_grid={ALLOWED['param_grid']}). \n"
    #             f"Valid parameters are: {list(ALLOWED.keys())}"
    #         )
    #
    #
    #     # set GSTCV params
    #     # GSTCV(Dask) parameters must be validated & set the long way
    #     for gstcv_param in gstcv_params:
    #         if gstcv_param not in ALLOWED_GSTCV_PARAMS:
    #             raise ValueError(
    #                 _invalid_est_param(gstcv_param, ALLOWED_GSTCV_PARAMS)
    #             )
    #         setattr(self, gstcv_param, params[gstcv_param])
    #
    #
    #     # set estimator params ** * ** * ** * ** * ** * ** * ** * ** * ** *
    #     # IF self.estimator is dask/sklearn est/pipe, THIS SHOULD HANDLE
    #     # EXCEPTIONS FOR INVALID PASSED PARAMS. Must set params on estimator,
    #     # not _estimator, because _estimator may not exist (until fit())
    #     try:
    #         self.estimator.set_params(**est_params)
    #     except TypeError:
    #         raise TypeError(f"estimator must be an instance, not the class")
    #     except AttributeError:
    #         raise
    #     except Exception as e:
    #         raise Exception(
    #             f'estimator.set_params() raised for reason other than TypeError '
    #             f'(estimator is class, not instance) or AttributeError (not an '
    #             f'estimator.) -- {e}'
    #         ) from None
    #
    #     # this is stop-gap validation in case an estimator (of a makeshift
    #     # sort, perhaps) does not block setting invalid params.
    #     for est_param in est_params:
    #         if est_param not in ALLOWED_EST_PARAMS:
    #             raise ValueError(
    #                 _invalid_est_param(est_param, ALLOWED_EST_PARAMS)
    #             )
    #     # END set estimator params ** * ** * ** * ** * ** * ** * ** * **
    #
    #     del ALLOWED_EST_PARAMS, ALLOWED_GSTCV_PARAMS, _invalid_est_param
    #
    #     return self


    def transform(self, X: Iterable[Iterable[Union[int, float]]]):

        """

        Call transform on the estimator with the best found parameters.
        Only available if refit is not False and the underlying estimator
        supports transform.


        Parameters
        ----------
        X:
            Iterable[Iterable[Union[int, float]]] - Must contain all
            numerics. Must be able to convert to a numpy.ndarray (GSTCV)
            or dask.array.core.Array (GSTCVDask). Must fulfill the input
            assumptions of the underlying estimator.


        Return
        ------
        -
            The best_estimator_ transform method result for X.


        """


        self.estimator_hasattr('transform')

        self.check_refit__if_false_block_attr('transform')

        self.check_is_fitted()

        _X, passed_feature_names = self._handle_X_y(X, y=None)[0::2]

        self._validate_features(passed_feature_names)

        with self._scheduler as scheduler:
            return self.best_estimator_.transform(_X)


    # END SKLEARN / DASK GSTCV Methods #################################
    ####################################################################


    ####################################################################
    # SUPPORT METHODS ##################################################

    def _validate_and_reset(self) -> None:

        """

        Validate common __init__ args/kwargs for GSTCV and GSTCVDask.


        Return
        ------
        -
            None

        """

        self._param_grid = _validate_thresholds__param_grid(
            self.thresholds,
            self.param_grid
        )

        self.scorer_ = _validate_scoring(self.scoring)
        self.multimetric_ = len(self.scorer_) > 1

        # VALIDATE refit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        self._refit = _validate_refit(self.refit, self.scorer_)

        # IF AN INSTANCE HAS ALREADY BEEN fit() WITH refit != False,
        # POST-REFIT ATTRS WILL BE AVAILABLE. BUT IF SUBSEQUENTLY refit
        # IS SET TO FALSE VIA set_params, THE POST-REFIT ATTRS NEED TO BE
        # REMOVED. IN THIS CONFIGURATION (i) THE REMOVE WILL HAPPEN AFTER
        # fit() IS CALLED AGAIN WHERE THIS METHOD (val&reset) IS CALLED
        # NEAR THE TOP OF fit() (ii) SETTING NEW PARAMS VIA set_params()
        # WILL LEAVE POST-REFIT ATTRS AVAILABLE ON AN INSTANCE THAT SHOWS
        # CHANGED PARAM SETTINGS (AS VIEWED VIA get_params()) UNTIL fit()
        # IS RUN.
        if self._refit is False:
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


    def estimator_hasattr(self, attr_or_method_name: str) -> None:

        """

        Check if in estimator has an attribute or method. If yes, return
        None. If not, raise AttributeError.


        Parameters
        ----------
        attr_or_method_name:
            str - the attribute or method name to look for.


        Return
        ------
        None


        """

        if not hasattr(self.estimator, attr_or_method_name):
            raise AttributeError(
                f"This '{type(self).__name__}' has no attribute"
                f" '{attr_or_method_name}'"
            )
        else:
            return


    def check_refit__if_false_block_attr(
            self,
            attr_or_method_name: str
    ) -> None:

        """

        Block attributes and methods that are not to be accessed if
        refit is False. If refit is False, raise AttributeError, if
        True, return None.


        Parameters
        ----------
        attr_or_method_name:
            str - the attribute or method name to block.


        Return
        ------
        -
            None

        """


        if not self.refit:
            raise AttributeError(
                f"This {type(self).__name__} instance was initialized with "
                f"`refit=False`. {attr_or_method_name} is available only after "
                f"refitting on the best parameters. You can refit an estimator "
                f"manually using the `best_params_` attribute"
            )
        else:
            return


    def check_is_fitted(self) -> None:

        """

        Block access to attributes and methods if instance is not fitted.
        If not fitted, raise Attribute Error, if fitted, return None.


        Return
        ------
        -
            None

        """

        if not hasattr(self, '_refit'):
            raise AttributeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using "
                f"this estimator."
            )
        else:
            return


    def _validate_features(self, passed_feature_names: np.ndarray) -> None:

        """

        Validate that feature names passed to a method via a dataframe
        have the exact names and order as those seen during fit, if fit
        was done with a dataframe. If not, raise ValueError. Return None
        if fit was done with an array or if the object passed to a
        method is an array.


        Parameters
        ----------
        passed_feature_names:
            NDArray[str] - shape (n_features, ), the column header from
            a dataframe passed to a method.


        Return
        ------
        -
            None



        """

        if passed_feature_names is None:
            return

        if not hasattr(self, 'feature_names_in_'):
            return

        pfn = passed_feature_names
        fni = self.feature_names_in_

        if np.array_equiv(pfn, fni):
            return
        else:

            base_err_msg = (f"The feature names should match those that "
                            f"were passed during fit. ")

            if len(pfn) == len(fni) and np.array_equiv(sorted(pfn), sorted(fni)):

                # when out of order
                # ValueError: base_err_msg
                # Feature names must be in the same order as they were in fit.
                addon = (f"\nFeature names must be in the same order as they "
                         f"were in fit.")

            else:

                addon = ''

                UNSEEN = []
                for col in pfn:
                    if col not in fni:
                        UNSEEN.append(col)
                if len(UNSEEN) > 0:
                    addon += f"\nFeature names unseen at fit time:"
                    for col in UNSEEN:
                        addon += f"\n- {col}"
                del UNSEEN

                SEEN = []
                for col in fni:
                    if col not in pfn:
                        SEEN.append(col)
                if len(SEEN) > 0:
                    addon += f"\nFeature names seen at fit time, yet now missing:"
                    for col in SEEN:
                        addon += f"\n- {col}"
                del SEEN

            message = base_err_msg + addon

            raise ValueError(message)

    # END SUPPORT METHODS ##############################################
    ####################################################################





















