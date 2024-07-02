# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from dask_ml.model_selection import KFold, GridSearchCV
from dask_ml.linear_model import LogisticRegression
import dask.array as da
import numpy as np
from dask_ml.datasets import make_classification

from dask.distributed import Client, LocalCluster



client = Client(n_workers=4, threads_per_worker=1)   #LocalCluster(n_workers=4, threads_per_worker=1))



clf = LogisticRegression()

X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0,
                           n_informative=5, n_classes=2, chunks=(200,5)
)




def KFoldGetter(splits, X, y):
    return KFold(n_splits=splits).split(X,y)


def fold_splitter(train_idxs, test_idxs, X, y):
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]


scores = []
for idx, (train_idxs, test_idxs) in enumerate(KFoldGetter(5,X,y)):
    print(f'tater gives {idx} shnewks')
    X_train, y_train, X_test, y_test = fold_splitter(train_idxs, test_idxs, X, y)
    clf.fit(X_train, y_train)#, scheduler=client)
    scores.append(clf.score(X_test, y_test))#, scheduler=client))

print(scores)











































