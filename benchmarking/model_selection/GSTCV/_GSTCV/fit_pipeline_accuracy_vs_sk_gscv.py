# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this module tests the equality of GSTCV's cv_results_ with
# 0.5 threshold against sklearn GSCV cv_results_ 100 times over
# for a pipeline.

# parallels tests/model_selection/GSTCV/GSTCV/fit/accuracy_test


import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV



good_estimator = \
    Pipeline(
        steps=[
            (
                'OneHot',
                OneHotEncoder(drop='first')
            ),
            (
                'Logistic',
                LogisticRegression(max_iter=10_000, solver='lbfgs')
            )
        ]
    )

good_param_grid = [
    {'OneHot__min_frequency': [10, 20], 'Logistic__C': [.1]},
    {'OneHot__min_frequency': [45, 55], 'Logistic__C': [.001]}
]

good_cv_int = 4

good_error_score = 'raise'

good_scorer = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'precision': precision_score,
    'recall': recall_score
}

_GSTCV = GSTCV(
    estimator=good_estimator,
    param_grid=good_param_grid,
    thresholds=[0.5],
    cv=good_cv_int,
    error_score=good_error_score,
    refit=False,
    verbose=0,
    scoring=good_scorer,
    n_jobs=-1,
    pre_dispatch='2*n_jobs',
    return_train_score=True
)

sk_gscv = GridSearchCV(
    good_estimator,
    good_param_grid,
    cv=good_cv_int,
    error_score=good_error_score,
    verbose=0,
    scoring={k: make_scorer(v) for k, v in good_scorer.items()},
    n_jobs=-1,
    return_train_score=True,
    refit=False
)

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

if __name__ == '__main__':

    for trial in range(100):
        print(f'running trial {trial}...', end='')

        _features=5
        _samples=1_000
        good_X = np.random.choice(list('abcdefghijklmnop'), (_samples, _features))
        good_y = np.random.randint(0, 2, _samples)


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        pd_gstcv_cv_results = pd.DataFrame(_GSTCV.fit(good_X, good_y).cv_results_)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        pd_sk_cv_results = pd.DataFrame(sk_gscv.fit(good_X, good_y).cv_results_)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        assert len(pd_gstcv_cv_results) == len(pd_sk_cv_results), \
            f"different rows in cv_results_"


        _ = pd_sk_cv_results.columns.to_numpy()
        _drop = [i for i in pd_gstcv_cv_results.columns if 'threshold' in i]
        __ = pd_gstcv_cv_results.drop(columns=_drop).columns.to_numpy()

        if not np.array_equiv(_, __):
            _max_len = max(map(len, (_, __)))
            DF = pd.DataFrame(index=range(_max_len), columns=['gstcv', 'gscv']).fillna('-')
            DF.iloc[:len(__), 0] = __
            DF.iloc[:len(_), 1] = _
            print(DF)
            raise AssertionError(f'columns not equal / out of order')

        for column in pd_gstcv_cv_results:

            if 'threshold' not in column and 'time' not in column:
                assert column in pd_sk_cv_results, \
                    print(f'\033[91mcolumn {column} not in!\033[0m')

            if 'threshold' in column:
                assert (pd_gstcv_cv_results[column] == 0.5).all()
                continue

            if 'time' in column:
                assert (pd_gstcv_cv_results[column] > 0).all()
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

                print(f'\033[92m')
                print(f'column {column} values OK!')
                print(f'gstcv[{column}] = {pd_gstcv_cv_results[column].to_numpy()}')
                print(f'sk_gscv[{column}] = {pd_sk_cv_results[column].to_numpy()}')
                print(f'\033[0m')

            except:
                # check param columns
                if not np.array_equiv(
                    pd_gstcv_cv_results[column][MASK].to_numpy(),
                    pd_sk_cv_results[column][MASK].to_numpy()
                ):

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

                    raise Exception(f'trial {trial} GSTCV != sk gscv')





