# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# 25_05_06_16_34_00 pizza this is on the block for permanent skip. this
# is almost
# completely redundant with the GSTCV test except for the accuracy of
# the numbers returned by the DaskGSTCV parallel modules. The GSTCV
# test confirms the GSTCV cv_results layout (except threshold columns)
# is identical to sk gscv layout (correct number of rows, correct column
# names in the correct order) and that the GSTCV parallel modules get
# numbers identical to sk. IF WE CAN INDEPENDENTLY CONFIRM THE ACCURACY
# OF THE GSTCVDASK PARALLEL MODULES RESULTS IN OTHER TESTS THEN THIS
# TEST CAN BE SKIPPED PERMANENTLY BUT KEEP THE FILE FOR THIS EXPLANATION.





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

from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask

pytest.skip(
    reason=f'pizza this test is on the block for permanent skip',
    allow_module_level=True
)

class TestFitAccuracy:

    # 24_07_10 this module tests the equality of GSTCVDask's cv_results_
    # with 0.5 threshold against sklearn GSCV cv_results_.
    # pizza decide what to do with the benchmarking file
    # there is a benchmarking file that passes the tests in freeform


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # create tight log init params to supersede session params for this test.
    # need tighter params on logistic to get GSTCV & sk_GSCV to agree.
    @staticmethod
    @pytest.fixture()
    def special_sk_log_init_params():
        return {
            'C': 1e-3,
            'tol': 1e-6, # need 1e-6 here to pass est accuracy tests
            'max_iter': 10000, # need 10000 here to pass est accuracy tests
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

        _sk_GSCV = sk_GSCV(**sk_gscv_init_params)
        _sk_GSCV.set_params(
            estimator=special_sk_est_log,
            param_grid=param_grid_sk_log,
            refit=False
        )
        return _sk_GSCV


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


    # dont use _client, too slow 24_08_26

    @pytest.mark.parametrize('_param_grid',
        (
            [
                {'C': [1e-1, 1e0], 'fit_intercept': [True, False]}
            ],
            [
                {'C': [1]},
                {'C': [1], 'fit_intercept': [False], 'solver': ['lbfgs']}
            ],
        )
    )
    @pytest.mark.parametrize('_return_train_score', (True, False))
    def test_accuracy_vs_sk_gscv(
        self, _param_grid, standard_cv_int, standard_error_score, _scorer,
        standard_cache_cv, standard_iid, _return_train_score, X_da, y_da,
        special_sk_est_log, special_sk_GSCV_est_log_one_scorer_prefit,
        _client  # 25_05_06 slightly faster with client
    ):

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        TestCls = GSTCVDask(
            estimator=special_sk_est_log,
            param_grid=_param_grid,
            thresholds=[0.5],
            cv=standard_cv_int,
            error_score=standard_error_score,
            refit=False,
            verbose=0,
            scoring=_scorer,
            cache_cv=standard_cache_cv,
            iid=standard_iid,
            return_train_score=_return_train_score
        )

        TestCls.fit(X_da, y_da)

        gstcv_cv_results = pd.DataFrame(TestCls.cv_results_)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        out_sk_gscv = special_sk_GSCV_est_log_one_scorer_prefit
        out_sk_gscv.set_params(
            param_grid=_param_grid,
            scoring={k: make_scorer(v) for k,v in _scorer.items()},
            return_train_score=_return_train_score
        )

        out_sk_gscv.fit(X_da.compute(), y_da.compute())

        sk_cv_results = pd.DataFrame(out_sk_gscv.cv_results_)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        assert gstcv_cv_results.shape[0] == sk_cv_results.shape[0], \
            f"different rows in cv_results_"

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _gscv_cols = sk_cv_results.columns.to_numpy()
        _drop = [i for i in gstcv_cv_results.columns if 'threshold' in i]
        _gstcv_cols = gstcv_cv_results.drop(columns=_drop).columns.to_numpy()
        del _drop
        assert np.array_equiv(_gscv_cols, _gstcv_cols), \
            f'columns not equal / out of order'
        del _gscv_cols, _gstcv_cols
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        for column in gstcv_cv_results:

            if 'threshold' in column:
                assert (gstcv_cv_results[column] == 0.5).all()
                continue  # notice continuing here
            elif 'time' in column:
                assert (gstcv_cv_results[column] > 0).all()
                assert (sk_cv_results[column] > 0).all()
                continue  # notice continuing here
            else:
                assert column in sk_cv_results, \
                    f'\033[91mcolumn {column} not in!\033[0m'
                # notice we flow thru to more tests

            # DONT CHECK GSTCVDask NUMBERS AGAINST sk_gscv. SK USES
            # StratifiedKFold & GSTCVDask USES dask KFold, WHICH GIVE
            # DIFFERENT SPLITS AND THEREFORE SLIGHTLY DIFFERENT SCORES

            # MASK = np.logical_not(gstcv_cv_results[column].isna())
            #
            # try:
            #     _gstcv_out = gstcv_cv_results[column][MASK]
            #     _gstcv_out = _gstcv_out.to_numpy(dtype=np.float64)
            #     _sk_out = sk_cv_results[column][MASK]
            #     _sk_out = _sk_out.to_numpy(dtype=np.float64)
            #     raise UnicodeError
            # except UnicodeError:
            #     # check floats
            #     assert np.allclose(_gstcv_out, _sk_out, atol=0.00001)
            # except:
            #     # check param columns
            #     assert np.array_equiv(
            #         gstcv_cv_results[column][MASK].to_numpy(),
            #         sk_cv_results[column][MASK].to_numpy()
            #     )



