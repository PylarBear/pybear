# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from dask_ml.datasets import make_classification
from dask_ml.linear_model import LogisticRegression
from xgboost.dask import XGBClassifier
from lightgbm.dask import LGBMClassifier

from pybear.utilities._benchmarking import time_memory_benchmark as tmb

# time trials linux 24_07_27
# lgt_clf       time = 9.941 +/- 0.038 sec; mem = 0.000 +/- 0.000 MB
# xgb_clf_1     time = 0.438 +/- 0.021 sec; mem = 0.200 +/- 0.400 MB
# xgb_clf_2     time = 1.530 +/- 0.013 sec; mem = 0.000 +/- 0.000 MB
# xgb_clf_3     time = 0.842 +/- 0.016 sec; mem = 0.000 +/- 0.000 MB
# xgb_clf_4     time = 0.393 +/- 0.004 sec; mem = 0.200 +/- 0.400 MB
# lgbm_clf      time = 0.443 +/- 0.026 sec; mem = 0.000 +/- 0.000 MB


X, y = make_classification(n_classes=2, n_features=5, n_samples=10_000,
                           n_repeated=0, n_redundant=0, chunks=(1_000,5))


lgt_clf = LogisticRegression(C=1e-3, tol=1e-6)
xgb_clf_1 = XGBClassifier(tree_method='auto', max_depth=3, learning_rate=0.1, n_estimators=100)
xgb_clf_2 = XGBClassifier(tree_method='exact', max_depth=3, learning_rate=0.1, n_estimators=100)
xgb_clf_3 = XGBClassifier(tree_method='approx', max_depth=3, learning_rate=0.1, n_estimators=100)
xgb_clf_4 = XGBClassifier(tree_method='hist', max_depth=3, learning_rate=0.1, n_estimators=100)
lgbm_clf = LGBMClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)



RESULTS = tmb(
    ('lgt_clf', lgt_clf.fit, [X, y], {}),
    # ('xgb_clf_1', xgb_clf_1.fit, [X, y], {}),
    # ('xgb_clf_2', xgb_clf_2.fit, [X, y], {}),
    # ('xgb_clf_3', xgb_clf_3.fit, [X, y], {}),
    ('xgb_clf_4', xgb_clf_4.fit, [X, y], {}),
    ('lgbm_clf', lgbm_clf.fit, [X, y], {}),
    rest_time=1,
    number_of_trials=7
)























