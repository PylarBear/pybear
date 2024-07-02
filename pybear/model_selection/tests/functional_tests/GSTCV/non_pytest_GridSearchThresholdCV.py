import numpy as np
import pandas as pd
from sklearn.datasets import make_classification as sk_make_classification
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
import dask.dataframe as ddf
from dask_ml.datasets import make_classification as da_make_classification
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
import string

from model_selection.GSTCV._GSTCV import GridSearchThresholdCV




test_package = 'sklearn'  # sklearn / dask
data_type = 'array'    # array / dataframe
_n_classes = 2
_rows, _columns = 1_000, 20    #7_000
column_names = list(string.ascii_lowercase[:24])[:_columns]





if test_package == 'sklearn':
    X, y = sk_make_classification(
        n_classes=_n_classes,
        n_samples=_rows,
        n_features=_columns,
        n_informative=_columns,
        n_redundant=0,
        weights=[0.75, 0.25]
    )

    if data_type.lower() == 'dataframe':
        X = pd.DataFrame(data=X, columns=column_names)
        # X = X['a']#.to_frame()
        y = pd.DataFrame(y)

    estimator = sklearn_LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-6,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=10000,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=-1,
        l1_ratio=None
    )

    scheduler = None

elif test_package == 'dask':
    X, y = da_make_classification(
        n_classes=_n_classes,
        n_samples=_rows,
        n_features=_columns,
        n_informative=_columns,
        n_redundant=0,
        weights=[0.9, 0.1],
        chunks=(_rows//10, _columns)
    )



    if data_type.lower() == 'dataframe':
        X = ddf.from_array(X, columns=column_names, chunksize=(_rows//10, _columns))
        y = ddf.from_array(y, chunksize=(_rows//10, _n_classes))

    estimator = dask_LogisticRegression(
        penalty='l2',
        # dual=False,                         # Ignored
        tol=1e-6,
        C=1.0,
        fit_intercept=True,
        # intercept_scaling=1.0,               # Ignored
        # class_weight=None,                   # Ignored
        random_state=None,
        solver='lbfgs',
        max_iter=10000,
        # multi_class='ovr',                    # Ignored
        # verbose,                              # Ignored
        # warm_start,                           # Ignored
        # n_jobs,                               # Ignored
        solver_kwargs=None
    )

    scheduler = None

else:
    raise Exception(f"Must specify test_package... must be 'sklearn' or 'dask'")

param_grid = [{'C': np.logspace(-3, 3, 7),'tol': np.logspace(-3, -1, 3)},
              {'C': np.logspace(-7, -4, 4), 'tol': np.logspace(-6, -4, 3)}]


def refit_fxn(DUM_CV_RESULTS):
    # DUM_DF = pd.DataFrame(DUM_CV_RESULTS)
    # [print(_) for _ in DUM_CV_RESULTS]
    # return DUM_DF.index[DUM_DF['rank_test_balanced_accuracy']==1][0]
    return 0

TestCls = GridSearchThresholdCV(
    estimator,
    param_grid,
    scoring=['accuracy', 'balanced_accuracy'],
    thresholds=np.linspace(0,1,21),
    n_jobs=-1,
    cv=3,
    refit='balanced_accuracy',
    verbose=10,
    error_score=np.nan,
    return_train_score=True,
    # OTHER POSSIBLE KWARGS FOR DASK SUPPORT
    iid=True,
    scheduler=scheduler,
    cache_cv=True,
)

TestCls.fit(X, y)

print(f'best index = {TestCls.best_index_}')
print(f'best score = {TestCls.best_score_}')
print(f'best params = {TestCls.best_params_}')
print(f'best threshold = {TestCls.best_threshold_}')
print(f'best estimator = {TestCls.best_estimator_}')

# DF = pd.DataFrame(TestCls.cv_results_)
# if os.name == 'posix':
#     DF.to_csv(r'/home/bear/Desktop/cv_results_dump.ods', index=False)
# elif os.name == 'nt':
#     DF.to_csv(r'c:\users\bill\desktop\cv_results_dump.csv', index=False)











# ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
ATTR_NAMES = [
    'cv_results_',
    'best_estimator_',
    'best_score_',
    'best_params_',
    'best_index_',
    'scorer_',
    'n_splits_'
]

for attr in ATTR_NAMES:
    if hasattr(TestCls, attr):
        print(f"has {attr} == True")
    else:
        print(f"\033[91m{attr} EXCEPTED \033[0m")

# END ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

# METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

NAMES = [
    'decision_function',
    'fit',
    'get_params',
    'inverse_transform',
    'predict',
    'predict_log_proba',
    'predict_proba',
    'score',
    'set_params',
    'transform',
    'visualize'
]


for meth in NAMES:
    if hasattr(TestCls, meth):
        print(f"has {meth} == True")
    else:
        print(f"\033[91m{meth} EXCEPTED \033[0m")
# END METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
















