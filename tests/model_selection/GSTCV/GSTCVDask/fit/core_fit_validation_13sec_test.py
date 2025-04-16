# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from dask_ml.model_selection import KFold as dask_KFold

from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression
)


from sklearn.metrics import precision_score, recall_score

from pybear.model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from pybear.model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit



class TestCoreFitValidation:

    # def _core_fit(
    #     _X: XDaskWIPType,
    #     _y: YDaskWIPType,
    #     _estimator: ClassifierProtocol,
    #     _cv_results: CVResultsType,
    #     _cv: Union[int, GenericKFoldType],
    #     _error_score: Union[int, float, Literal['raise']],
    #     _verbose: int,
    #     _scorer: ScorerWIPType,
    #     _cache_cv: bool,
    #     _iid: bool,
    #     _return_train_score: bool,
    #     _PARAM_GRID_KEY: npt.NDArray[np.uint8],
    #     _THRESHOLD_DICT: dict[int, npt.NDArray[np.float64]],
    #     **params
    #     ) -> CVResultsType


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def helper_for_cv_results_and_PARAM_GRID_KEY(standard_cv_int, good_SCORER):
        param_grid = [
            {'C': [100, 1000, 10000]},
            {'solver': ['saga', 'lbfgs', 'sag']}
        ]

        out_cv_results, out_key = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            param_grid=param_grid,
            cv=standard_cv_int,
            scorer=good_SCORER,
            return_train_score=True
        )

        return out_cv_results, out_key


    @staticmethod
    @pytest.fixture
    def good_cv_results(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[0]


    @staticmethod
    @pytest.fixture
    def good_cv_arrays(X_da, y_da, standard_cv_int):
        return dask_KFold(n_splits=standard_cv_int).split(X_da, y_da)


    @staticmethod
    @pytest.fixture
    def good_SCORER():
        return {'precision': precision_score, 'recall': recall_score}


    @staticmethod
    @pytest.fixture
    def good_PARAM_GRID_KEY(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[1]


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
    def test_rejects_junk_X(self, junk_X, y_da, dask_est_log, good_cv_results,
        standard_cv_int, standard_error_score, good_SCORER, standard_cache_cv,
        standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT, _client
    ):

        with pytest.raises(TypeError):
            _core_fit(
                junk_X,
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
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(self, X_da, junk_y, dask_est_log, good_cv_results,
        standard_cv_int, standard_error_score, good_SCORER, standard_cache_cv,
        standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT, _client
    ):

        with pytest.raises(TypeError):
            _core_fit(
                X_da,
                junk_y,
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
                good_THRESHOLD_DICT
            )


    @pytest.mark.parametrize('junk_estimator',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_estimator(self, X_da, y_da, junk_estimator,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AttributeError):
            _core_fit(
                X_da,
                y_da,
                junk_estimator,
                good_cv_results,
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


    @pytest.mark.parametrize('bad_estimator', (dask_LinearRegression(),))
    def test_rejects_bad_estimator(self, X_da, y_da, bad_estimator,
        good_cv_results, standard_cv_int, standard_error_score, good_SCORER,
        standard_cache_cv, standard_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT,
        _client
    ):

        with pytest.raises(AttributeError):
            _core_fit(
                X_da,
                y_da,
                bad_estimator,
                good_cv_results,
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













