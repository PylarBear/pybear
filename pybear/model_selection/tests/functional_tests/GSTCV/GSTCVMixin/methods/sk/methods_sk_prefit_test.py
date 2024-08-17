# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV






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


    # sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def sk_est():
        return sk_LogisticRegression(
            C=1e-3,
            tol=1e-4,
            fit_intercept=False,
            solver='lbfgs'
        )


    @staticmethod
    @pytest.fixture
    def sk_GSTCV(sk_est, param_grid):

        def foo(refit, scoring):

            return GSTCV(
                estimator=sk_est,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                error_score='raise'
            )

        return foo


    @staticmethod
    @pytest.fixture
    def X_np():
        return np.random.randint(0, 10, (100, 5))


    @staticmethod
    @pytest.fixture
    def X_df(X_np):
        return pd.DataFrame(X_np)


    @staticmethod
    @pytest.fixture
    def y_np():
        return np.random.randint(0, 2, (100,))

    # END sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


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
                f"parameters. You can refit an estimator manually using the "
                f"`best_params_` attribute")

        return foo

    # END exception matches ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # END FIXTURES ### ### ### ### ### ### ### ### ### ### ### ### ###
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###







    def test_methods(
        self, sk_GSTCV, _refit, _scoring, _format, X_np, X_df, y_np,
        _no_attribute, _not_fitted, _no_refit
    ):

        X_sk = X_np if _format == 'array' else X_df

        kwargs = {'refit': _refit, 'scoring': _scoring}

        GSTCV = sk_GSTCV(**kwargs)

        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'decision_function')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'decision_function')(X_sk)

        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(GSTCV, 'get_metadata_routing')()

        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCV', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'inverse_transform')(X_sk)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict')(X_sk)
        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_log_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict_log_proba')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict_log_proba')(X_sk)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict_proba')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict_proba')(X_sk)

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'score')(X_sk, y_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'score')(X_sk, y_np)

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCV', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'score_samples')(X_sk)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _no_attribute('GSTCV', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'transform')(X_sk)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "'GSTCV' object has no attribute 'visualize'"
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















