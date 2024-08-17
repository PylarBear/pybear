# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit





# 24_07_10 this module tests the equality of SK GSTCV's cv_results_ with
# 0.5 threshold against sklearn GSCV cv_results_.
class TestCoreFitAccuracy:

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
    def X_y_helper():


        return make_classification(
            n_classes=2,
            n_samples=1000,
            n_features=5,
            n_repeated=0,
            n_redundant=0,
            n_informative=5,
            shuffle=False
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
        return LogisticRegression(
            max_iter=10_000,
            solver='lbfgs',
            tol=1e-6
    )


    @staticmethod
    @pytest.fixture
    def good_cv_int():
        return 5


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


    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    # this is just an example, the columns are not correct for these test!
    # 'mean_fit_time'
    # 'std_fit_time'
    # 'mean_score_time'
    # 'std_score_time'
    # 'param_C'
    # 'param_solver'
    # 'params'
    # 'best_threshold_accuracy'
    # 'split0_test_accuracy'
    # 'split1_test_accuracy'
    # 'mean_test_accuracy'
    # 'std_test_accuracy'
    # 'rank_test_accuracy'
    # 'split0_train_accuracy'
    # 'split1_train_accuracy'
    # 'mean_train_accuracy'
    # 'std_train_accuracy'
    # 'best_threshold_balanced_accuracy'
    # 'split0_test_balanced_accuracy'
    # 'split1_test_balanced_accuracy'
    # 'mean_test_balanced_accuracy'
    # 'std_test_balanced_accuracy'
    # 'rank_test_balanced_accuracy'
    # 'split0_train_balanced_accuracy'
    # 'split1_train_balanced_accuracy'
    # 'mean_train_balanced_accuracy'
    # 'std_train_balanced_accuracy'




    @pytest.mark.parametrize('_scorer',
        (
            {
                 'precision': precision_score,
                 'recall': recall_score,
                 'accuracy': accuracy_score,
                 'balanced_accuracy': balanced_accuracy_score
            },
            {
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            }
        )
    )
    @pytest.mark.parametrize('_param_grid',
        (
            [
                {'C': [1e-1, 1e0, 1e1], 'fit_intercept': [True, False]}
            ],
            [
                {'C': [1e-2, 1e-1, 1e0], 'fit_intercept': [True, False]},
                {'C': [1e1, 1e2, 1e3], 'fit_intercept': [True, False]}
            ],
            [
                {'C': [1]},
                {'C': [1], 'fit_intercept': [False]},
                {'C': [1], 'fit_intercept': [False], 'solver': ['lbfgs']}
            ],
        )
    )
    @pytest.mark.parametrize('_n_jobs', (-1, 1))  # <==== 1 is important
    @pytest.mark.parametrize('_return_train_score', (True, False))
    def test_accuracy_vs_sk_gscv(self, good_X, good_y, good_estimator,
        good_cv_int, good_error_score, _scorer,  _n_jobs, _return_train_score,
        _param_grid):


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        good_cv_results, PARAM_GRID_KEY = _cv_results_builder(
            param_grid=_param_grid,
            cv=good_cv_int,
            scorer=_scorer,
            return_train_score=_return_train_score
        )

        gstcv_cv_results = _core_fit(
            good_X,
            good_y,
            good_estimator,
            good_cv_results,
            good_cv_int,
            good_error_score,
            0, # good_verbose,
            _scorer,
            _n_jobs,
            _return_train_score,
            PARAM_GRID_KEY,
            {i: np.array([0.5]) for i in range(len(_param_grid))} # THRESHOLD_DICT
        )

        pd_gstcv_cv_results = pd.DataFrame(gstcv_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        out_sk_gscv = GridSearchCV(
            good_estimator,
            _param_grid,
            cv=good_cv_int,
            error_score=good_error_score,
            verbose=0,
            scoring={k: make_scorer(v) for k,v in _scorer.items()},
            n_jobs=_n_jobs,
            return_train_score=_return_train_score,
            refit=list(_scorer.keys())[0]
        )

        out_sk_gscv.fit(good_X, good_y)

        sk_cv_results = out_sk_gscv.cv_results_

        pd_sk_cv_results = pd.DataFrame(sk_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



        assert len(pd_gstcv_cv_results) == len(pd_sk_cv_results), \
            f"different rows in cv_results_"


        _ = pd_gstcv_cv_results.to_numpy()
        MASK = list(map(lambda x: 'threshold' in x, pd_gstcv_cv_results.columns))
        _drop = pd_gstcv_cv_results.columns[MASK]
        __ = pd_gstcv_cv_results.drop(columns=_drop).columns.to_numpy()

        assert np.array_equiv(__, pd_sk_cv_results.columns), \
            f'columns not equal / out of order'
        del MASK, __

        for column in pd_gstcv_cv_results:

            if 'threshold' not in column and 'time' not in column:
                assert column in pd_sk_cv_results, \
                    print(f'\033[91mcolumn {column} not in!\033[0m')

            if 'threshold' in column:
                assert (pd_gstcv_cv_results[column] == 0.5).all()
                continue

            if 'time' in column:
                assert (pd_gstcv_cv_results[column] > 0).all()
                assert (gstcv_cv_results[column] > 0).all()
                continue


            MASK = np.logical_not(pd_gstcv_cv_results[column].isna())

            try:
                _gstcv_out = pd_gstcv_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )
                _sk_out = pd_sk_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )

                raise UnicodeError

            except UnicodeError:
                # check floats
                assert np.allclose(_gstcv_out, _sk_out, atol=0.00001)

            except:
                # check param columns
                assert np.array_equiv(
                    pd_gstcv_cv_results[column][MASK].to_numpy(),
                    pd_sk_cv_results[column][MASK].to_numpy()
                )




























