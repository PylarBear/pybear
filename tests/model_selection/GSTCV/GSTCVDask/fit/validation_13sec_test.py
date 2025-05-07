# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from sklearn.linear_model import LinearRegression as sk_LinearRegression

from sklearn.metrics import (
    precision_score,
    recall_score
)

from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask



class TestFitValidation:


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def good_SCORER():
        return {'precision': precision_score, 'recall': recall_score}


    @staticmethod
    @pytest.fixture(scope='function')
    def base_gstcv_dask(
        sk_est_log, param_grid_sk_log, standard_cv_int,
        standard_error_score, good_SCORER, standard_cache_cv, standard_iid
    ):
        # dont overwrite a session fixture with new params!

        return GSTCVDask(
            estimator=sk_est_log,
            param_grid=param_grid_sk_log,
            thresholds=np.linspace(0,1,11),
            cv=standard_cv_int,
            error_score=standard_error_score,
            verbose=10,
            scoring=good_SCORER,
            refit=False,
            cache_cv=standard_cache_cv,
            iid=standard_iid,
            n_jobs=-1,
            return_train_score=True,
            scheduler=None
        )

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    # test validation * * ** * * ** * * ** * * ** * * ** * * ** * * ** *

    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, junk_X, y_da, base_gstcv_dask, _client):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            base_gstcv_dask.fit(junk_X, y_da)


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(self, X_da, junk_y, base_gstcv_dask, _client):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, junk_y)


    @pytest.mark.parametrize('junk_estimator',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_estimator(
        self, X_da, y_da, junk_estimator, base_gstcv_dask, _client
    ):
        # dont use set_params here
        base_gstcv_dask.estimator=junk_estimator

        with pytest.raises(AttributeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('bad_estimator', (sk_LinearRegression(),))
    def test_rejects_bad_estimator(
        self, X_da, y_da, bad_estimator, base_gstcv_dask, _client
    ):

        base_gstcv_dask.set_params(estimator=bad_estimator)

        with pytest.raises(AttributeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_param_grid',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_param_grid(
        self, X_da, y_da, junk_param_grid, base_gstcv_dask, _client
    ):

        base_gstcv_dask.set_params(param_grid=junk_param_grid)

        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_thresholds',
        (-1, 3.14, True, False, 'trash', min, ['a', 'b'], ('a', 'b'),
         {'a', 'b'}, lambda x: x)
    )
    def test_rejects_junk_thresholds(
        self, X_da, y_da, base_gstcv_dask, junk_thresholds, _client
    ):

        base_gstcv_dask.set_params(thresholds=junk_thresholds)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('bad_thresholds', ({'a': 1}, {0: 1}, {0: 'b'}))
    def test_rejects_bad_thresholds(
        self, X_da, y_da, base_gstcv_dask, bad_thresholds, _client
    ):

        base_gstcv_dask.set_params(thresholds=bad_thresholds)

        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_cv',
        (-1, 0, 1, 3.14, [0, 1], (0, 1), {0, 1}, True, False, 'trash', min,
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cv(
        self, X_da, y_da, base_gstcv_dask, junk_cv, _client
    ):

        base_gstcv_dask.set_params(cv=junk_cv)

        with pytest.raises((ValueError, TypeError, AssertionError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_error_score',
        (True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_error_score(
        self, X_da, y_da, base_gstcv_dask, junk_error_score, _client
    ):

        base_gstcv_dask.set_params(error_score=junk_error_score)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_verbose',
        (-10, -1, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_verbose(
        self, X_da, y_da, base_gstcv_dask, junk_verbose, _client
    ):

        base_gstcv_dask.set_params(verbose=junk_verbose)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_refit',
        (-10, -1, True, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_refit(
        self, X_da, y_da, base_gstcv_dask, junk_refit, _client
    ):

        base_gstcv_dask.set_params(refit=junk_refit)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_scoring',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1),
         {'a': 1}, {0: 1}, {'trash': 'junk'}, lambda x: x)
    )
    def test_rejects_junk_scoring(
        self, X_da, y_da, base_gstcv_dask, junk_scoring, _client
    ):

        base_gstcv_dask.set_params(scoring=junk_scoring)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_n_jobs',
        (-2, 0, 3.14, True, False, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_n_jobs(
        self, X_np, y_np, base_gstcv_dask, junk_n_jobs
    ):

        base_gstcv_dask.set_params(n_jobs=junk_n_jobs)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv_dask.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_cache_cv',
        (-2, 0, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cache_cv(
        self, X_da, y_da, base_gstcv_dask, junk_cache_cv, _client
    ):

        base_gstcv_dask.set_params(cache_cv=junk_cache_cv)

        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_iid',
        (-2, 0, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
        {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_iid(
        self, X_da, y_da, base_gstcv_dask, junk_iid, _client
    ):

        base_gstcv_dask.set_params(iid=junk_iid)

        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, y_da)


    @pytest.mark.parametrize('junk_return_train_score',
        (-1, 0, 1, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_return_train_score(
        self, X_da, y_da, base_gstcv_dask, junk_return_train_score, _client
    ):

        base_gstcv_dask.set_params(return_train_score=junk_return_train_score)

        with pytest.raises(TypeError):
            base_gstcv_dask.fit(X_da, y_da)


    # END test validation * * ** * * ** * * ** * * ** * * ** * * ** * *







