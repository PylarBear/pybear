# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
from sklearn.datasets import make_classification as sk_make_classification
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV as sk_GSCV
from model_selection.GSTCV._GSTCV.GSTCV import GSTCV



if __name__ == '__main__':

    X_np, y = sk_make_classification(
        n_samples=10_000, n_features=5, n_redundant=0,
        n_informative=5, n_classes=2
    )

    clf = sk_LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-6,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=-1
    )

    param_grid = {
        'C': np.logspace(-5,-4,2),
        # 'solver': ['lbfgs', 'saga'],
        # 'thresholds': np.linspace(0,1,101)
    }


    # gstcv = GSTCV(
    #     clf,
    #     param_grid,
    #     thresholds=np.linspace(0,1,5),
    #     scoring=['accuracy'], #'balanced_accuracy'],
    #     n_jobs=1,
    #     cv=3,
    #     refit='accuracy',
    #     error_score='raise',
    #     return_train_score=False
    # )
    #
    # gstcv.fit(X_np, y)
    #
    # print(gstcv.predict(X_np))
    # print(type(gstcv.predict(X_np)))



    gscv = sk_GSCV(
        clf,
        param_grid,
        # thresholds=[0.5],
        scoring='balanced_accuracy',
        n_jobs=1,
        cv=3,
        refit='balanced_accuracy',
        error_score='raise',
        return_train_score=False
    )


    gscv.fit(X_np, y)

    print(f'gscv multimetric_ = {gscv.multimetric_}')

    # print(gscv.best_params_)
















