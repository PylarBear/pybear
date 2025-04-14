# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd

from dask_ml.model_selection import KFold as dask_KFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from pybear.model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from pybear.model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit

# 24_08_11 this module tests the dask GSTCV operation of:
# 1) the cache_cv kwarg, proves the equality of cv_results_ when cache_cv
# is True or False.
# 2) the cv kwarg, proves the equality of cv_results_ when cv as int and
# cv as iterable are expected to give identical folds.
# 3) the iid kwarg, proves the equality of cv_results_ on 2 independent
# calls to GSTCV with iid = False on the same data and same cv. iid = True
# cannot be tested for equality with iid = False, because the different
# sampling of train and test will cause different scores.



class TestCVCacheCVIid:

    # def _core_fit(
    #     _X: XDaskWIPType,
    #     _y: YDaskWIPType,
    #     _estimator: ClassifierProtocol,
    #     _cv_results: CVResultsType,
    #     _cv: Union[int, Iterable[GenericKFoldType]],
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
    def good_param_grid():
        return [
            {'C': [1e-5], 'fit_intercept': [True]},
            {'C': [1e-1], 'fit_intercept': [False]}
        ]


    @staticmethod
    @pytest.fixture
    def helper_for_cv_results_and_PARAM_GRID_KEY(
        standard_cv_int, good_scorer, good_param_grid
    ):

        out_cv_results, out_key = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            param_grid=good_param_grid,
            cv=standard_cv_int,
            scorer=good_scorer,
            return_train_score=True
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
    def good_iid():
        return False


    @staticmethod
    @pytest.fixture
    def good_PARAM_GRID_KEY(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[1]


    @staticmethod
    @pytest.fixture
    def good_THRESHOLD_DICT():
        return {0: np.linspace(0,1,5), 1: np.linspace(0,1,3)}

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    # dont pass client, too slow 25_04_13
    def test_accuracy(self, X_da, y_da, dask_est_log,
        good_cv_results, standard_cv_int, standard_error_score, good_scorer,
        good_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT): #, _client):

        # test equivalent cv as int or iterable give same output

        out_true = _core_fit(
            X_da,
            y_da,
            dask_est_log,
            good_cv_results,
            standard_cv_int,  # <====== cv
            standard_error_score,
            0, # good_verbose,
            good_scorer,
            True,   # <====== cache_cv
            good_iid,  # <====== iid
            True, # good_return_train_score,
            good_PARAM_GRID_KEY,
            good_THRESHOLD_DICT
        )

        out_false = _core_fit(
            X_da,
            y_da,
            dask_est_log,
            good_cv_results,
            dask_KFold(n_splits=standard_cv_int).split(X_da, y_da), # <== cv
            standard_error_score,
            0, # good_verbose,
            good_scorer,
            False,   # <====== cache_cv
            good_iid,   # <===== iid
            True, # good_return_train_score,
            good_PARAM_GRID_KEY,
            good_THRESHOLD_DICT
        )


        assert pd.DataFrame(data=out_true).equals(pd.DataFrame(data=out_false))

        # cv_results_ being equal for both outs proves that comparable
        # cv as int & cv as iterator give same output, cache_cv True and
        # False give the same output, and successive independent calls
        # on the same data & splits with iid = False give the same output






























