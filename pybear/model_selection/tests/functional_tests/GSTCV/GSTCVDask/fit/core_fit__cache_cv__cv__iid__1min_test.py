# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd


from dask_ml.datasets import make_classification as dask_make_classification
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression

from dask_ml.model_selection import KFold as dask_KFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit



# 24_08_11 this module tests the dask GSTCV operation of:
# 1) the cache_cv kwarg, proves the equality of cv_results_ when cache_cv
# is True or False.
# 2) the cv kwarg, proves the equality of cv_results_ when cv as int and
# cv as iterable are expected to give identical folds.
# 3) the iid kwarg, proves the equality of cv_results_ on 2 independent
# calls to GSTCV with iid = False on the same data and same cv. iid = True
# cannot be tested for equality with iid = False, because the different
# sampling of train and test will cause different scores.
#
# These only needs to be tested for an estimator, as the caching,
# splitting, and iid sampling processes are independent of estimator
# being a single estimator or a pipeline.

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
    @pytest.fixture(scope='module')
    def X_y_helper():
        _samples = 100
        _features = 3
        return dask_make_classification(
            n_classes=2,
            n_samples=_samples,
            n_features=_features,
            n_repeated=0,
            n_redundant=0,
            n_informative=5,
            shuffle=False,
            chunks=(_samples, _features)
    )


    @staticmethod
    @pytest.fixture
    def good_X(X_y_helper):
        return X_y_helper[0]


    @staticmethod
    @pytest.fixture
    def good_y(X_y_helper):
        return X_y_helper[1]


    @staticmethod
    @pytest.fixture
    def good_estimator():
        return dask_LogisticRegression(max_iter=10_000, solver='lbfgs', tol=1e-6)


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
        good_cv_int, good_scorer, good_param_grid
    ):

        out_cv_results, out_key = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            param_grid=good_param_grid,
            cv=good_cv_int,
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
    def good_cv_int():
        return 4


    @staticmethod
    @pytest.fixture
    def good_error_score():
        return 'raise'


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
        return {0: np.linspace(0,1,21), 1: np.linspace(0,1,11)}


    # @staticmethod
    # @pytest.fixture(scope='module')
    # def _client():
    #     client = distributed.Client(n_workers=1, threads_per_worker=1)
    #     yield client
    #     client.close()

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def test_accuracy_cache_cv(self, good_X, good_y, good_estimator,
        good_cv_results, good_cv_int, good_error_score,good_scorer,
        good_iid, good_PARAM_GRID_KEY, good_THRESHOLD_DICT): #, _client):

        # test equivalent cv as int or iterable give same output

        out_true = _core_fit(
            good_X,
            good_y,
            good_estimator,
            good_cv_results,
            good_cv_int,  # <====== cv
            good_error_score,
            0, # good_verbose,
            good_scorer,
            True,   # <====== cache_cv
            good_iid,  # <====== iid
            True, # good_return_train_score,
            good_PARAM_GRID_KEY,
            good_THRESHOLD_DICT
        )

        out_false = _core_fit(
            good_X,
            good_y,
            good_estimator,
            good_cv_results,
            list(dask_KFold(n_splits=good_cv_int).split(good_X, good_y)),   # <====== cv
            good_error_score,
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






























