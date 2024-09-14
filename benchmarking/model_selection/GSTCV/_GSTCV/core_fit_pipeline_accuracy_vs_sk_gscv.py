# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd


from sklearn.model_selection import (
    GridSearchCV,
    KFold
)
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

from pybear.model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from pybear.model_selection.GSTCV._GSTCV._fit._core_fit import _core_fit

# 24_07_10 this module tests the equality of GSTCV's cv_results_ with
# 0.5 threshold against sklearn GSCV cv_results_ 100 times over.



for trial in range(100):
    print(f'running trial {trial}...', end='')

    _features=5
    _samples=1_000
    good_X = np.random.choice(list('abcdefghijklmnop'), (_samples, _features))
    good_y = np.random.randint(0, 2, _samples)


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

    good_cv_iter = list(KFold(n_splits=good_cv_int).split(good_X, good_y))

    good_error_score = 'raise'

    good_scorer = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    }

    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    good_cv_results, good_PARAM_GRID_KEY = _cv_results_builder(
        # DO NOT PUT 'thresholds' IN PARAM GRIDS!
        param_grid=good_param_grid,
        cv=good_cv_int,
        scorer=good_scorer,
        return_train_score=True
    )

    gstcv_cv_results = _core_fit(
        good_X,
        good_y,
        good_estimator,
        good_cv_results,
        good_cv_iter,
        good_error_score,
        0,  # good_verbose,
        good_scorer,
        -1,  # good_n_jobs,
        True,  # good_return_train_score,
        good_PARAM_GRID_KEY,
        {i: np.array([0.5]) for i in range(len(good_param_grid))}  # good_THRESHOLD_DICT
    )

    pd_gstcv_cv_results = pd.DataFrame(gstcv_cv_results)
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    out_sk_gscv = GridSearchCV(
        good_estimator,
        good_param_grid,
        cv=good_cv_iter,
        error_score=good_error_score,
        verbose=0,
        scoring={k: make_scorer(v) for k, v in good_scorer.items()},
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

    if not np.array_equiv(_, __):
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









