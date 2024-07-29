# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LogisticRegression as sklearn_Logistic

import dask.array as da
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_Logistic

from pybear.model_selection import GSTCV, GSTCVDask

from pybear.utils import time_memory_benchmark as tmb


X_np = np.random.randint(0, 10, (500, 5))
y_np = np.random.randint(0, 2, (500,))
X_da = da.array(X_np).rechunk((100, 5))
y_da = da.array(y_np).rechunk((100,))


def skGSCV(X, y, C=10):
    skGSCV = sklearn_GridSearchCV(
        estimator=sklearn_Logistic(),
        param_grid={'C':[C]},
        scoring=['balanced_accuracy','accuracy'],
        refit='balanced_accuracy',
        return_train_score=True
    )

    skGSCV.fit(X,y)

    return skGSCV


def skGSTCV(X, y, C=10):
    skGSTCV = GSTCV(
        estimator=sklearn_Logistic(),
        param_grid={'C': [C]},
        scoring=['balanced_accuracy', 'accuracy'],
        refit='balanced_accuracy',
        thresholds=np.linspace(0, 1, 21),
        return_train_score=True
    )

    skGSTCV.fit(X, y)

    return skGSTCV


def daGSCV(X, y, C=10):
    daGSCV = dask_GridSearchCV(
        estimator=dask_Logistic(),
        param_grid={'C':[C]},
        scoring=['balanced_accuracy','accuracy'],
        refit='balanced_accuracy',
        return_train_score=True
    )

    daGSCV.fit(X,y)

    return daGSCV


def daGSTCV(X, y, C=10):
    daGSTCV = GSTCVDask(
        estimator=dask_Logistic(),
        param_grid={'C': [C]},
        scoring=['balanced_accuracy', 'accuracy'],
        refit='balanced_accuracy',
        thresholds=np.linspace(0, 1, 21),
        return_train_score=True
    )

    daGSTCV.fit(X, y)

    return daGSTCV




if __name__ == '__main__':

    tmb(
        ('skGSCV_.001', skGSCV, [X_np, y_np], {'C':.001}),
        ('skGSCV_1000', skGSCV, [X_np, y_np], {'C':1000}),
        ('skGSTCV_.001', skGSTCV, [X_np, y_np], {'C':.001}),
        ('skGSTCV_1000', skGSTCV, [X_np, y_np], {'C':1000}),
        ('daGSCV_.001', daGSCV, [X_da, y_da], {'C': .001}),
        ('daGSCV_1000', daGSCV, [X_da, y_da], {'C': 1000}),
        ('daGSTCV_.001', daGSTCV, [X_da, y_da], {'C': .001}),
        ('daGSTCV_1000', daGSTCV, [X_da, y_da], {'C': 1000}),
        rest_time=1,
        number_of_trials=3
    )











