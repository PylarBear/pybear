import numpy as np
import pandas as pd
from sklearn.datasets import make_classification as sk_make_classification
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
import dask.dataframe as ddf
from dask_ml.datasets import make_classification as da_make_classification
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
import string

from model_selection.GSTCV.GSTCV import GridSearchThresholdCV




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

else: raise Exception(f"Must specify test_package... must be 'sklearn' or 'dask'")

param_grid = [{'C': np.logspace(-3, 3, 7),'tol': np.logspace(-3,-1,3)},
              {'C': np.logspace(-7, -4, 4), 'tol': np.logspace(-6,-4,3)}]


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

quit()

# TEST FOR cv_results_builder ######################################################################################
# PIZZA THIS WAS TESTED 24_02_11_17_54_00 AND SHOULDNT NEED TO BE "FIXED"
# 24_02_13_14_29_00 SO MUCH FOR THAT, PROBABLY NEED TO ADD A best_threshold COLUMN
print(f'RUNNING cv_results_builder TEST')

param_grid = [
                {'kernel': ['rbf'], 'gamma': [0.1, 0.2], 'test_param': [1, 2, 3]},
                {'kernel': ['poly'], 'degree': [2, 3], 'test_param': [1, 2, 3]},
]

correct_cv_results_len = np.sum(list(map(np.prod, [[len(_) for _ in __] for __ in map(dict.values, param_grid)])))


CV = [3,4,5]

SCORING = [['accuracy'], ['accuracy', 'balanced_accuracy']]

RETURN_TRAIN = [True, False]

UNIQUE_PARAMS = list({'param_' + _ for __ in param_grid for _ in __})

test_permutations = np.prod(list(map(len, (CV, SCORING, RETURN_TRAIN))))
print(f'number of test permutations = {test_permutations}\n')

ctr = 0
for _cv in CV:
    for _scoring in SCORING:
        for return_train in RETURN_TRAIN:
            ctr += 1
            print(f'\033[92mRunning test permutation number {ctr} of {test_permutations}...\033[0m')
            # BUILD VERIFICATION STUFF #########################################################################
            COLUMN_CHECK = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
            COLUMN_CHECK += UNIQUE_PARAMS
            COLUMN_CHECK += ['params']
            for sub_scoring in _scoring:
                if len(_scoring)==1: COLUMN_CHECK += [f'best_threshold']
                else: COLUMN_CHECK += [f'best_threshold_{sub_scoring}']

                suffix = 'score' if len(_scoring)==1 else sub_scoring
                for split in range(_cv):
                    COLUMN_CHECK += [f'split{split}_test_{suffix}']
                COLUMN_CHECK += [f'mean_test_{suffix}']
                COLUMN_CHECK += [f'std_test_{suffix}']
                COLUMN_CHECK += [f'rank_test_{suffix}']
                if return_train:
                    for split in range(_cv):
                        COLUMN_CHECK += [f'split{split}_train_{suffix}']
                    COLUMN_CHECK += [f'mean_train_{suffix}']
                    COLUMN_CHECK += [f'std_train_{suffix}']

            # BUILD VERIFICATION STUFF #########################################################################

            # RUN cv_results_builder AND GET CHARACTERISTICS ###################################################
            cv_results_output = cv_results_builder(param_grid, _cv, _scoring, return_train)
            OUTPUT_COLUMNS = list(cv_results_output.keys())
            OUTPUT_LEN = len(cv_results_output['mean_fit_time'])
            # RUN cv_results_builder AND GET CHARACTERISTICS ###################################################

            # COMPARE OUTPUT TO CONTROLS ########################################################################
            for out_col in OUTPUT_COLUMNS:
                if out_col not in COLUMN_CHECK:
                    raise Exception(f"\033[91m{out_col} is in OUTPUT_COLUMNS but not in COLUMN_CHECK\033[0m")
            for check_col in COLUMN_CHECK:
                if check_col not in OUTPUT_COLUMNS:
                    raise Exception(f"\033[91m{check_col} is in COLUMN_CHECK but not in OUTPUT_COLUMNS\033[0m")

            output_len = len(cv_results_output['mean_fit_time'])
            if output_len != correct_cv_results_len:
                raise Exception(f"\033[91moutput rows ({output_len}) does not equal expected rows ({correct_cv_results_len})\033[0m")

            del output_len

            print(f'\033[92mTrial {ctr} passed all tests\033[0m')

            # COMPARE OUTPUT TO CONTROLS ########################################################################

# END TEST FOR cv_results_builder ######################################################################################













# ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
ATTR_NAMES = [
                'cv_results_',
                'TestCls.best_estimator_',
                'TestCls.best_score_',
                'TestCls.best_params_',
                'TestCls.best_index_',
                'TestCls.scorer_',
                'TestCls.n_splits_'
]

ATTRS = [
                TestCls.cv_results_,
                TestCls.best_estimator_,
                TestCls.best_score_,
                TestCls.best_params_,
                TestCls.best_index_,
                TestCls.scorer_,
                TestCls.n_splits_
]

for name, attr in zip(ATTR_NAMES, ATTRS):
    try: print(f"{name} == {attr}")
    except: print(f"\033[91m{name} EXCEPTED \033[0m")

# END ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

# METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

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

METHODS = [
            TestCls.decision_function(X),
            TestCls.fit(X, y),
            TestCls.get_params(True),
            TestCls.inverse_transform(Xt),
            TestCls.predict(X),
            TestCls.predict_log_proba(X),
            TestCls.predict_proba(X),
            TestCls.score(X, y),
            TestCls.set_params(y),
            TestCls.transform(X),
            TestCls.visualize(filename=None, format=None)
]

for name, attr in zip(ATTR_NAMES, ATTRS):
    try: print(f"{name} == {attr}")
    except: print(f"\033[91m{name} EXCEPTED \033[0m")
# END METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

