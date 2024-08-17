# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np
import dask.array as da
import dask.dataframe as ddf
import distributed
# from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from xgboost.dask import DaskXGBClassifier as dask_XGBC
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask



@pytest.mark.parametrize('_format', ('array', 'df'))
@pytest.mark.parametrize('_scoring', ('accuracy', ['accuracy', 'balanced_accuracy']))
@pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
class TestGSTCVDaskMethodsPostFit:

    # methods & signatures (besides fit)
    # --------------------------
    # decision_function(X)
    # inverse_transform(Xt)
    # predict(X)
    # predict_log_proba(X)
    # predict_proba(X)
    # score(X, y=None, **params)
    # score_samples(X)
    # transform(X)
    # visualize(filename=None, format=None)



    @staticmethod
    @pytest.fixture
    def param_grid():
        return {'n_estimators': [100], 'max_depth': [5]}
        # return {'C': [1e-4], 'tol': [1e-4]}


    # dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def dask_est():

        return dask_XGBC(
            n_estimators=50,
            max_depth=5,
            tree_method='hist'
        )

        # return dask_LogisticRegression(
        #     C=1e-3,
        #     tol=1e-4,
        #     max_iter=100,
        #     fit_intercept=False,
        #     solver='lbfgs'
        # )


    @staticmethod
    @pytest.fixture
    def dask_GSTCV(dask_est, param_grid):

        def foo(refit, scoring):

            return GSTCVDask(
                estimator=dask_est,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                error_score='raise'
            )

        return foo


    @staticmethod
    @pytest.fixture
    def X_da():
        return da.random.randint(0, 10, (100, 5))


    @staticmethod
    @pytest.fixture
    def X_ddf(X_da):
        return ddf.from_dask_array(X_da)


    @staticmethod
    @pytest.fixture
    def y_da():
        return da.random.randint(0, 2, (100,))


    @staticmethod
    @pytest.fixture
    def _client():

        client = distributed.Client(n_workers=None, threads_per_worker=1)
        yield client
        client.close()


    # END dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # exception matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture
    def _no_attribute():
        def foo(_gscv_type, _method):
            return f"This '{_gscv_type}' has no attribute '{_method}'"

        return foo


    @staticmethod
    @pytest.fixture
    def _no_refit():
        def foo(_object, _apostrophes: bool, _method):
            if _apostrophes:
                __ = "`refit=False`"
            else:
                __ = "refit=False"

            return (f"This {_object} instance was initialized with {__}. "
                f"{_method} is available only after refitting on the best "
                f"parameters. You can refit an estimator manually using the "
                f"`best_params_` attribute")

        return foo

    # END exception matches ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # END FIXTURES ### ### ### ### ### ### ### ### ### ### ### ### ###
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


    def test_methods(
        self, dask_GSTCV, _refit, _scoring, _format, X_da, X_ddf, y_da,
        _no_attribute, _no_refit, _client
    ):

        X_dask = X_da if _format == 'array' else X_ddf

        kwargs = {'refit': _refit, 'scoring': _scoring}

        GSTCV = dask_GSTCV(**kwargs).fit(X_dask, y_da)

        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'decision_function')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'decision_function')(X_dask)

        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(GSTCV, 'get_metadata_routing')()

        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'inverse_transform')(X_dask)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict')(X_dask)

        elif _refit == 'accuracy':

            __ = getattr(GSTCV, 'predict')(X_dask)
            assert isinstance(__, da.core.Array)
            assert __.dtype == np.uint8

        elif callable(_refit):

            # this is to accommodate lack of threshold when > 1 scorer
            if isinstance(_scoring, list) and len(_scoring) > 1:
                with pytest.raises(AttributeError):
                    getattr(GSTCV, 'predict')(X_dask)
            else:
                __ = getattr(GSTCV, 'predict')(X_dask)
                assert isinstance(__, da.core.Array)
                assert __.dtype == np.uint8


        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'predict_log_proba')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'predict_log_proba')(X_dask)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict_proba')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):

            __ = getattr(GSTCV, 'predict_proba')(X_dask)
            assert isinstance(__, da.core.Array)
            assert __.dtype == np.float32

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'score')(X_dask, y_da)

        elif _refit == 'accuracy':

            __ = getattr(GSTCV, 'score')(X_dask, y_da)
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1

        elif callable(_refit):
            __ = getattr(GSTCV, 'score')(X_dask, y_da)
            if not isinstance(_scoring, list) or len(_scoring) == 1:
                # if refit fxn & one scorer, score is always returned
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1
            else:
                # if refit fxn & >1 scorer, refit fxn is returned
                assert callable(__)
                cvr = GSTCV.cv_results_
                assert isinstance(__(cvr), int) # refit(cvr) returns best_index_
                # best_index_ must be in range of the rows in cvr
                assert __(cvr) >= 0
                assert __(cvr) < len(cvr[list(cvr)[0]])

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'score_samples')(X_dask)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'transform')(X_dask)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "visualize is not implemented in GSTCVDask"
        with pytest.raises(NotImplementedError, match=exc_info):
            getattr(GSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















