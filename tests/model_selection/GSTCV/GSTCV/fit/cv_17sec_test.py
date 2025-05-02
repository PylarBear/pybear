# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from pybear.model_selection.GSTCV._GSTCVMixin._fit._cv_results._cv_results_builder \
    import _cv_results_builder

from pybear.model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit



# 24_08_11 this module tests the operation of the cv kwarg in sk GSTCV,
# proves the equality of cv_results_ when cv as int and cv as iterable
# are expected to give identical folds. This only needs to be tested for
# an estimator, as the splitting processes is independent of estimator
# being a single estimator or a pipeline.
class TestCV:

    # def _core_fit(
    #     _X: XSKWIPType,
    #     _y: YSKWIPType,
    #     _estimator: ClassifierProtocol,
    #     _cv_results: CVResultsType,
    #     _cv: Union[int, GenericKFoldType],
    #     _error_score: Union[int, float, Literal['raise']],
    #     _verbose: int,
    #     _scorer: ScorerWIPType,
    #     _n_jobs: Union[int, None],
    #     _return_train_score: bool,
    #     _PARAM_GRID_KEY: npt.NDArray[np.uint8],
    #     _THRESHOLD_DICT: dict[int, npt.NDArray[np.float64]],
    #     **params
    #     ) -> CVResultsType



    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    @staticmethod
    @pytest.fixture
    def good_param_grid():
        # keep only one dict in here (or go change THRESHOLD_DICT)
        return [
            {'C': [1e-5, 1e-4], 'fit_intercept': [True]}
        ]


    @staticmethod
    @pytest.fixture
    def helper_for_cv_results_and_PARAM_GRID_KEY(
        standard_cv_int, good_scorer, good_param_grid
    ):

        out_cv_results, out_key = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            _param_grid=good_param_grid,
            _cv=standard_cv_int,
            _scorer=good_scorer,
            _return_train_score=True
        )

        return out_cv_results, out_key


    @staticmethod
    @pytest.fixture
    def good_cv_results(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[0]


    @staticmethod
    @pytest.fixture
    def good_scorer():
        return {
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score
        }


    @staticmethod
    @pytest.fixture
    def good_PARAM_GRID_KEY(helper_for_cv_results_and_PARAM_GRID_KEY):
        return helper_for_cv_results_and_PARAM_GRID_KEY[1]


    @staticmethod
    @pytest.fixture
    def good_THRESHOLD_DICT(good_param_grid):
        return {idx: np.linspace(0,1,11) for idx in range(len(good_param_grid))}

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    @pytest.mark.parametrize('_n_jobs', (-1, 1))  # 1 is important
    def test_accuracy_cv_int_vs_cv_iter(
        self, X_np, y_np, sk_est_log, good_cv_results, standard_cv_int,
        standard_error_score, good_scorer, good_PARAM_GRID_KEY,
        good_THRESHOLD_DICT, _n_jobs
    ):

        # test equivalent cv as int or iterable give same output

        out_int = _core_fit(
            X_np,
            y_np,
            sk_est_log,
            good_cv_results,
            standard_cv_int,   # <===============
            standard_error_score,
            0, # good_verbose,
            good_scorer,
            _n_jobs,
            '2*n_jobs',
            True, # good_return_train_score,
            good_PARAM_GRID_KEY,
            good_THRESHOLD_DICT
        )

        out_iter = _core_fit(
            X_np,
            y_np,
            sk_est_log,
            good_cv_results,
            KFold(n_splits=standard_cv_int).split(X_np, y_np),   # <=========
            standard_error_score,
            0, # good_verbose,
            good_scorer,
            _n_jobs,
            '2*n_jobs',
            True, # good_return_train_score,
            good_PARAM_GRID_KEY,
            good_THRESHOLD_DICT
        )

        assert pd.DataFrame(data=out_int).equals(pd.DataFrame(data=out_iter))































