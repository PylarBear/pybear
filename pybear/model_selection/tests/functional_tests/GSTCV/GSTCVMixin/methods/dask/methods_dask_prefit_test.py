# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import dask.array as da
import dask.dataframe as ddf

from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask






@pytest.mark.parametrize('_format', ('array', 'df'))
@pytest.mark.parametrize('_scoring', ('accuracy', ['accuracy', 'balanced_accuracy']))
@pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
class TestGSCVMethodsPreFit:

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
        return {'C': [1e-4], 'tol': [1e-4]}


    # dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def dask_est():
        return dask_LogisticRegression(
            C=1e-3,
            tol=1e-4,
            max_iter=100,
            fit_intercept=False,
            solver='lbfgs'
        )


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
        return da.random.randint(0, 10, (100, 5)).rechunk((10, 5))


    @staticmethod
    @pytest.fixture
    def X_ddf(X_da):
        return ddf.from_dask_array(X_da)


    @staticmethod
    @pytest.fixture
    def y_da():
        return da.random.randint(0, 2, (100,)).rechunk((10,))

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
    def _not_fitted():
        def foo(_object):
            return (f"This {_object} instance is not fitted yet. Call 'fit' "
                f"with appropriate arguments before using this estimator.")

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
                f"parameters.")

        return foo


    # END exception matches ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # END FIXTURES ### ### ### ### ### ### ### ### ### ### ### ### ###
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###







    def test_methods(
        self, dask_GSTCV, _refit, _scoring, _format, X_da, X_ddf, y_da,
        _no_attribute, _not_fitted, _no_refit
    ):

        X_dask = X_da if _format == 'array' else X_ddf

        kwargs = {'refit': _refit, 'scoring': _scoring}

        DaskGSTCV = dask_GSTCV(**kwargs)

        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'decision_function')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'decision_function')(X_dask)

        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(DaskGSTCV, 'get_metadata_routing')()

        # END get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'inverse_transform')(X_dask)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'predict')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'predict')(X_dask)
        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'predict_log_proba')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'predict_log_proba')(X_dask)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'predict_proba')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'predict_proba')(X_dask)

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'score')(X_dask, y_da)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'score')(X_dask, y_da)

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'score_samples')(X_dask)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCVDask', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'transform')(X_dask)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _not_fitted('GSTCVDask')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















