# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.metrics import (
    precision_score,
    recall_score
)

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV



class TestFitValidation:


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def good_SCORER():
        return {'precision': precision_score, 'recall': recall_score}


    @staticmethod
    @pytest.fixture(scope='function')
    def base_gstcv(
        sk_est_log, param_grid_sk_log, standard_cv_int,
        standard_error_score, good_SCORER
    ):
        # dont overwrite a session fixture with new params!

        return GSTCV(
            estimator=sk_est_log,
            param_grid=param_grid_sk_log,
            thresholds=np.linspace(0,1,11),
            cv=standard_cv_int,
            error_score=standard_error_score,
            verbose=10,
            scoring=good_SCORER,
            refit=False,
            n_jobs=-1,
            pre_dispatch='2*n_jobs',
            return_train_score=True
        )

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    # test validation * * ** * * ** * * ** * * ** * * ** * * ** * * ** *

    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, junk_X, y_np, base_gstcv):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            base_gstcv.fit(junk_X, y_np)


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(self, X_np, junk_y, base_gstcv):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, junk_y)


    @pytest.mark.parametrize('junk_estimator',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_estimator(
        self, X_np, y_np, junk_estimator, base_gstcv
    ):
        # dont use set_params here
        base_gstcv.estimator=junk_estimator

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('bad_estimator', (LinearRegression(), Ridge()))
    def test_rejects_bad_estimator(
        self, X_np, y_np, bad_estimator, base_gstcv
    ):

        base_gstcv.set_params(estimator=bad_estimator)

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)



    @pytest.mark.parametrize('junk_param_grid',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         lambda x: x)
    )
    def test_rejects_junk_param_grid(
        self, X_np, y_np, base_gstcv, junk_param_grid
    ):

        base_gstcv.set_params(param_grid=junk_param_grid)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_thresholds',
        (-1, 3.14, True, False, 'trash', min, ['a', 'b'], ('a', 'b'),
         {'a', 'b'}, lambda x: x)
    )
    def test_rejects_junk_thresholds(
        self, X_np, y_np, base_gstcv, junk_thresholds
    ):

        base_gstcv.set_params(thresholds=junk_thresholds)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('bad_thresholds', ({'a': 1}, {0: 1}, {0: 'b'}))
    def test_rejects_bad_thresholds(
        self, X_np, y_np, base_gstcv, bad_thresholds
    ):

        base_gstcv.set_params(thresholds=bad_thresholds)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_cv',
        (-1, 0, 1, 3.14, [0, 1], (0, 1), {0, 1}, True, False, 'trash', min,
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cv(
        self, X_np, y_np, base_gstcv, junk_cv
    ):

        base_gstcv.set_params(cv=junk_cv)

        with pytest.raises((ValueError, TypeError, AssertionError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_error_score',
        (True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_error_score(
        self, X_np, y_np, base_gstcv, junk_error_score
    ):

        base_gstcv.set_params(error_score=junk_error_score)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_verbose',
        (-10, -1, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_verbose(
        self, X_np, y_np, base_gstcv, junk_verbose
    ):

        base_gstcv.set_params(verbose=junk_verbose)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_refit',
        (-1, 0, 1, 3.14, True, 'trash', min, [0, 1], (0, 1),
         {'a': 1}, {0: 1}, {'trash': 'junk'}, lambda x: x)
    )
    def test_rejects_junk_refit(
        self, X_np, y_np, base_gstcv, junk_refit
    ):

        base_gstcv.set_params(refit=junk_refit)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_scoring',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1),
         {'a': 1}, {0: 1}, {'trash': 'junk'}, lambda x: x)
    )
    def test_rejects_junk_scoring(
        self, X_np, y_np, base_gstcv, junk_scoring
    ):

        base_gstcv.set_params(scoring=junk_scoring)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_n_jobs',
        (-2, 0, 3.14, True, False, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_n_jobs(
        self, X_np, y_np, base_gstcv, junk_n_jobs
    ):

        base_gstcv.set_params(n_jobs=junk_n_jobs)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_pre_dispatch',
        (-2, 0, False, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_pre_dispatch(
        self, X_np, y_np, base_gstcv, junk_pre_dispatch
    ):

        base_gstcv.set_params(pre_dispatch=junk_pre_dispatch)

        # this is raised by joblib, let it raise whatever
        with pytest.raises(Exception):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_return_train_score',
        (-1, 0, 1, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_return_train_score(
        self, X_np, y_np, base_gstcv, junk_return_train_score
    ):

        base_gstcv.set_params(return_train_score=junk_return_train_score)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)

    # END test validation * * ** * * ** * * ** * * ** * * ** * * ** * *







