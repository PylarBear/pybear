# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from dask_ml.model_selection import KFold

from sklearn.linear_model import LinearRegression as sk_LinearRegression



from sklearn.metrics import (
    precision_score,
    recall_score
)

from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


pytest.skip(reason='pizza says so', allow_module_level=True)


class TestFitValidation:


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def good_cv_arrays(X_da, y_da, standard_cv_int):
        return KFold(n_splits=standard_cv_int).split(X_da, y_da)


    @staticmethod
    @pytest.fixture
    def good_SCORER():
        return {'precision': precision_score, 'recall': recall_score}


    @staticmethod
    @pytest.fixture
    def good_THRESHOLD_DICT():
        return {0: np.linspace(0,1,21), 1: np.linspace(0,1,11)}

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    # test validation * * ** * * ** * * ** * * ** * * ** * * ** * * ** *

    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, junk_X, y_da, dask_GSTCV_est_log_one_scorer_prefit):

        # this is raised by dask_ml.KFold, let it raise whatever
        with pytest.raises(UnicodeError):
            dask_GSTCV_est_log_one_scorer_prefit.fit(junk_X, y_da)


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(self, X_da, junk_y, dask_GSTCV_est_log_one_scorer_prefit):

        # this is raised by _get_kfold validation
        with pytest.raises(UnicodeError):
            dask_GSTCV_est_log_one_scorer_prefit.fit(X_da, junk_y)


    @pytest.mark.parametrize('junk_estimator',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_estimator(
        self, X_da, y_da, junk_estimator, param_grid_dask_log, standard_cv_int,
        standard_error_score, good_SCORER, dask_GSTCV_est_log_one_scorer_prefit,
        _client
    ):

        # dont overwrite a session fixture with new params!

        with pytest.raises(AttributeError):
            GSTCVDask(
                estimator=junk_estimator,
                param_grid=param_grid_dask_log,
                cv=standard_cv_int,
                error_score=standard_error_score,
                verbose=10,
                scoring=good_SCORER,
                cache_cv=standard_cache_cv,
                iid=standard_iid,
                return_train_score=True
            ).fit(X_da, y_da)


    @pytest.mark.parametrize('bad_estimator', (sk_LinearRegression(),))
    def test_rejects_bad_estimator(self, X_da, y_da, bad_estimator,
        dask_log_init_params,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AttributeError):
            GSTCVDask(
                estimator=bad_estimator,
                param_grid=dask_log_init_params,
                cv=standard_cv_int,
                error_score=standard_error_score,
                verbose=10,
                scoring=good_SCORER,
                cache_cv=standard_cache_cv,
                iid=standard_iid,
                return_train_score=True
            ).fit(X_da, y_da)


    @pytest.mark.parametrize('junk_cv_results',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         lambda x: x)
    )
    def test_rejects_junk_cv_results(self, X_da, y_da, dask_est_log,
        junk_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(TypeError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                junk_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )



    @pytest.mark.parametrize('bad_cv_results', ({'a': 1}, {'params': 1}))
    def test_rejects_bad_cv_results(self, X_da, y_da, dask_est_log,
        bad_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises((KeyError, TypeError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                bad_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_cv_int',
        (-1, 0, 1, 3.14, [0, 1], (0, 1), {0, 1}, True, False, None, 'trash', min,
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cv(self, X_da, y_da, dask_est_log,
        good_cv_results, junk_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises((ValueError, TypeError, AssertionError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                junk_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_error_score',
        (True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_error_score(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, junk_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises((TypeError, AssertionError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                junk_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_verbose',
        (-10, -1, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_verbose(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, junk_verbose,
        good_SCORER, standard_cache_cv, standard_iid, good_PARAM_GRID_KEY,
        good_THRESHOLD_DICT, _client
    ):

        with pytest.raises((TypeError, AssertionError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                junk_verbose,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_SCORER',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1),
         {'a': 1}, {0: 1}, {'trash': 'junk'}, lambda x: x)
    )
    def test_rejects_junk_SCORER(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, junk_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AssertionError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                junk_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_cache_cv',
        (-2, 0, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cache_cv(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        junk_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AssertionError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                junk_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_iid',
        (-2, 0, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_iid(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, junk_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AssertionError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                junk_iid,
                True,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_return_train_score',
        (-1, 0, 1, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_return_train_score(self, X_da, y_da,
        dask_est_log, good_cv_results, standard_cv_int, standard_error_score,
        good_SCORER, standard_cache_cv, standard_iid, junk_return_train_score,
        good_PARAM_GRID_KEY, good_THRESHOLD_DICT, _client
    ):

        with pytest.raises(AssertionError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                junk_return_train_score,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_PARAM_GRID_KEY',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [99, 100], (99, 100),
         {100, 101}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_PARAM_GRID_KEY(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, junk_PARAM_GRID_KEY,
        good_THRESHOLD_DICT, _client
    ):

        with pytest.raises((AssertionError, TypeError, ValueError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                junk_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_THRESHOLD_DICT',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         lambda x: x)
    )
    def test_rejects_junk_THRESHOLD_DICT(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        junk_THRESHOLD_DICT, _client
    ):

        with pytest.raises((TypeError, AssertionError)):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                junk_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('bad_THRESHOLD_DICT', ({'a': 1}, {0: 1}, {0: 'b'}))
    def test_rejects_bad_THRESHOLD_DICT(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        bad_THRESHOLD_DICT, _client
    ):

        with pytest.raises(AssertionError):
            _core_fit(
                X_da,
                y_da,
                dask_est_log,
                good_cv_results,
                standard_cv_int,
                standard_error_score,
                10,
                good_SCORER,
                standard_cache_cv,
                standard_iid,
                True,
                good_PARAM_GRID_KEY,
                bad_THRESHOLD_DICT
            )



    # END test validation * * ** * * ** * * ** * * ** * * ** * * ** * *







