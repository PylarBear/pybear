# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import distributed

from dask_ml.datasets import make_classification as dask_make_classification

from sklearn.model_selection import ParameterGrid

from dask_ml.model_selection import (
    GridSearchCV as dask_GridSearchCV,
    KFold as dask_KFold
)
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score
)

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder import \
    _cv_results_builder

from model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit

# 24_07_10 this module tests the equality of GSTCV's cv_results_ with
# 0.5 threshold against sklearn GSCV cv_results_.


if __name__ == '__main__':

    with distributed.Client(n_workers=4, threads_per_worker=1):
        _features=5
        _samples=100_000
        good_X, good_y = dask_make_classification(
            n_classes=2,
            n_samples=_samples,
            n_features=_features,
            n_repeated=0,
            n_redundant=0,
            n_informative=5,
            shuffle=True,
            chunks=((_samples//10, _features)),
            random_state=7
        )

        good_estimator = dask_LogisticRegression(
            max_iter=10_000,
            solver='lbfgs',
            random_state=69
        )


        # 24_07_11 there is an anomaly in SK GridSearch where passing


        good_param_grid = [
            {'C': [1]},
            # {'C': [1], 'solver':['lbfgs'], 'fit_intercept': [False]},
            # {'C': [1], 'fit_intercept': [True]},
            {'C': [1], 'fit_intercept': [False]}
            # {'fit_intercept': [True, False]},
        ]

        # 'C': [1e-1, 1e0, 1e1],
        # 'C': [1],
        # , 'solver': ['lbfgs']
        # 'solver': ['saga', 'lbfgs', 'sag'],

        good_cv_int = 4

        good_cv_iter = list(dask_KFold(n_splits=good_cv_int).split(good_X, good_y))

        good_error_score = 'raise'

        good_scorer = {
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score
        }

        good_cache_cv = True
        good_iid = True

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        good_cv_results, good_PARAM_GRID_KEY = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            param_grid=good_param_grid,
            cv=good_cv_int,
            scorer=good_scorer,
            return_train_score=True
        )

        param_grid = ParameterGrid(good_param_grid)
        good_cv_results['params'] = np.ma.masked_array(param_grid)
        del param_grid

        gstcv_cv_results = _core_fit(
            good_X,
            good_y,
            good_estimator,
            good_cv_results,
            good_cv_iter,
            good_error_score,
            0,  # good_verbose,
            good_scorer,
            good_cache_cv,
            good_iid,
            True,  # good_return_train_score,
            good_PARAM_GRID_KEY,
            {i: np.array([0.5]) for i in range(len(good_param_grid))}  # good_THRESHOLD_DICT
        )
        pd_gstcv_cv_results = pd.DataFrame(gstcv_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        _scorer = {k: make_scorer(v) for k, v in good_scorer.items()}

        out_sk_gscv = dask_GridSearchCV(
            good_estimator,
            good_param_grid,
            cv=good_cv_iter,
            error_score=good_error_score,
            cache_cv=good_cache_cv,
            iid=good_iid,
            scoring=_scorer,
            n_jobs=-1,
            return_train_score=True,
            refit=False
        )

        out_sk_gscv.fit(good_X, good_y)

        sk_cv_results = out_sk_gscv.cv_results_

        pd_sk_cv_results = pd.DataFrame(sk_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        assert len(pd_gstcv_cv_results) == len(pd_sk_cv_results), \
            f"different rows in cv_results_"


        _ = pd_sk_cv_results.columns.to_numpy()
        MASK = list(map(lambda x: 'threshold' in x, pd_gstcv_cv_results.columns))
        _drop = pd_gstcv_cv_results.columns[MASK]
        __ = pd_gstcv_cv_results.drop(columns=_drop).columns.to_numpy()
        # dask GSCV cv_results_ table order is different than SK, so sort
        if not np.array_equiv(sorted(_), sorted(__)):
            _max_len = max(map(len, (_, __)))
            DF = pd.DataFrame(index=range(_max_len), columns=['gstcv', 'gscv']).fillna('-')
            DF.iloc[:len(__), 0] = __
            DF.iloc[:len(_), 1] = _
            print(DF)
            raise AssertionError(f'columns not equal / out of order')

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

            OK = False

            MASK = np.logical_not(pd_gstcv_cv_results[column].isna())

            try:
                _gstcv_out = pd_gstcv_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )
                _sk_out = pd_sk_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )

                assert (_gstcv_out > 0).any()
                assert (_sk_out > 0).any()

                OK =  np.allclose(_gstcv_out, _sk_out, atol=0.00001)

            except:
                OK = np.array_equiv(
                    pd_gstcv_cv_results[column][MASK].to_numpy(),
                    pd_sk_cv_results[column][MASK].to_numpy()
                )

            if OK:
                print(f'\033[92m')
                print(f'column {column} values OK!')
                print(f'gstcv[{column}] = {pd_gstcv_cv_results[column].to_numpy()}')
                print(f'sk_gscv[{column}] = {pd_sk_cv_results[column].to_numpy()}')
                print(f'\033[0m')
            else:
                print(f'\033[91m')
                print(f'column {column} values wrong')
                print(f'gstcv[{column}] = {pd_gstcv_cv_results[column].to_numpy()}')
                print(f'sk_gscv[{column}] = {pd_sk_cv_results[column].to_numpy()}')

                if 'rank' in column:
                    print()
                    _scoring = 'balanced_accuracy' if 'balanced' in column else 'accuracy'
                    print(f"gstcv[f'mean_test_{_scoring}'] = \
                        {pd_gstcv_cv_results[f'mean_test_{_scoring}'].to_numpy()}")
                    print(f"sk_gscv[f'mean_test_{_scoring}'] = \
                        {pd_sk_cv_results[f'mean_test_{_scoring}'].to_numpy()}")
                    print('\033[0m')











