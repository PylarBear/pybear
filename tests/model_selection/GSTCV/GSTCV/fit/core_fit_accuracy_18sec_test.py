# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV as sk_GSCV
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from pybear.model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit
from pybear.model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder


class TestCoreFitAccuracy:

    # 24_07_10 this module tests the equality of SK GSTCV's cv_results_ with
    # 0.5 threshold against sklearn GSCV cv_results_.


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

    # create tight log init params to supersede session params for this test.
    # need tighter params on logistic to get _core_fit & sk_GSCV to agree.
    @staticmethod
    @pytest.fixture()
    def special_sk_log_init_params():
        return {
            'C': 1e-3,
            'tol': 1e-6, # need 1e-6 here to pass est/pipe accuracy tests
            'max_iter': 10000, # need 10000 here to pass est/pipe accuracy tests
            'fit_intercept': False,
            'solver': 'lbfgs'
        }


    @staticmethod
    @pytest.fixture
    def special_sk_est_log(special_sk_log_init_params):
        return sk_LogisticRegression(**special_sk_log_init_params)


    @staticmethod
    @pytest.fixture()
    def special_sk_GSCV_est_log_one_scorer_prefit(
            sk_gscv_init_params, special_sk_est_log, param_grid_sk_log
    ):

        __ = sk_GSCV(**sk_gscv_init_params)
        __.set_params(
            estimator=special_sk_est_log,
            param_grid=param_grid_sk_log,
            refit=False
        )
        return __


    @staticmethod
    @pytest.fixture
    def _scorer():
        return {
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score
        }

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    @pytest.mark.parametrize('_param_grid',
        (
            [
                {'C': [1e-1, 1e0], 'fit_intercept': [True, False]}
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
    def test_accuracy_vs_sk_gscv(self, _param_grid, standard_cv_int,
        standard_error_score, _scorer,  _n_jobs, _return_train_score,
        X_np, y_np, special_sk_est_log, special_sk_GSCV_est_log_one_scorer_prefit):


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        good_cv_results, PARAM_GRID_KEY = _cv_results_builder(
            param_grid=_param_grid,
            cv=standard_cv_int,
            scorer=_scorer,
            return_train_score=_return_train_score
        )

        gstcv_cv_results = _core_fit(
            X_np,
            y_np,
            special_sk_est_log,
            good_cv_results,
            standard_cv_int,
            standard_error_score,
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

        out_sk_gscv = special_sk_GSCV_est_log_one_scorer_prefit
        out_sk_gscv.set_params(
            param_grid=_param_grid,
            scoring={k: make_scorer(v) for k,v in _scorer.items()},
            n_jobs=_n_jobs,
            return_train_score=_return_train_score
        )

        out_sk_gscv.fit(X_np, y_np)

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
                    f'\033[91mcolumn {column} not in!\033[0m'

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




























