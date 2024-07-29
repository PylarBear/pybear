# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import time

from dask_ml.model_selection import KFold
from dask_ml.linear_model import LogisticRegression
from dask_ml.datasets import make_classification

from dask.distributed import Client



# repeated fits of a dask classifier under a dask KFold for-loop







clf = LogisticRegression(solver='lbfgs')

X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0,
                           n_informative=5, n_classes=2, chunks=(200,5)
)




def KFoldGetter(splits, X, y):
    return KFold(n_splits=splits).split(X,y)


def fold_splitter(train_idxs, test_idxs, X, y):
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]





if __name__ == '__main__':



    with Client(n_workers=4, threads_per_worker=1):
        scores = []
        for idx, (train_idxs, test_idxs) in enumerate(KFoldGetter(5,X,y)):
            t0 = time.time()
            print(f'start pass {idx}')
            print(f'type(train_idxs) = {type(train_idxs)}')
            X_train, y_train, X_test, y_test = fold_splitter(train_idxs, test_idxs, X, y)
            print(f'finished fold_splitter')
            clf.fit(X_train, y_train)
            print(f'finished fit, start score')
            scores.append(clf.score(X_test, y_test))
            print(f'time = {time.time() - t0}')

    print(scores)











































