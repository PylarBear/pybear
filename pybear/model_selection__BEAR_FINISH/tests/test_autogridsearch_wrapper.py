import pytest

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV

from pybear.model_selection__BEAR_FINISH import autogridsearch_wrapper



# BEAR FINISH TEST



DaskAutoGridSearch = autogridsearch_wrapper(dask_GridSearchCV)
SKLearnAutoGridSearch = autogridsearch_wrapper(sklearn_GridSearchCV)

# TEST demo() ########################################################################################################
estimator = 'dummy_estimator'

numerical_params = {
    # 'C1': [[3,4,5,6], [4,4,4,4,4,4], 'soft_float'],
    # 'C2': [[3, 4, 5, 6], [4, 4, 4, 4, 4, 4], 'hard_float'],
    # 'C3': [[3, 4, 5, 6], [4, 4, 4, 4, 4, 4], 'fixed_float'],
    # 'C4': [[3, 4, 5, 6], [4, 4, 4, 4, 4, 4], 'soft_integer'],
    # 'C5': [[3, 4, 5, 6], [4, 4, 4, 4, 4, 4], 'hard_integer'],
    # 'C6': [[3, 4, 5, 6], [4, 4, 4, 4, 4, 4], 'fixed_integer'],
    'C7': ['logspace', 0, 6, [7, 4, 4, 4, 4, 4], 'soft_float'],
    # 'C8': ['logspace', 0, 6, [7, 4, 4, 4, 4, 4], 'hard_float'],
    # 'C9': ['logspace', 1, 4, [4, 4, 4, 4, 4, 4], 'fixed_float'],
    # 'C10': ['logspace', 3, 6, [4, 4, 4, 4, 4, 4], 'soft_integer'],
    # 'C11': ['logspace', 3, 6, [4, 4, 4, 4, 4, 4], 'hard_integer'],
    # 'C12': ['logspace', 3, 6, [4, 4, 4, 4, 4, 4], 'fixed_integer'],
    # 'C13': ['linspace', 2.5, 9.5, [8,4,4,4,4,4], 'soft_float'],
    # 'C14': ['linspace', 2.5, 9.5, [8, 4, 4, 4, 4, 4], 'hard_float'],
    # 'C15': ['linspace', 2.5, 5.5, [4, 4, 4, 4, 4, 4], 'fixed_float'],
    # 'C16': ['linspace', 1, 7, [7,4,4,4,4,4], 'soft_integer'],
    # 'C17': ['linspace', 1, 7, [7, 3, 3, 3, 3, 3], 'hard_integer'],
    # 'C18': ['linspace', 1, 4, [4, 4, 4, 4, 4, 4], 'fixed_integer'],
    'l1ratio_1': ['linspace', 0, 1, [5, 4, 4, 4, 4, 4], 'hard_float'],
    # 'l1ratio_2': ['linspace', 0, 1, [2, 2, 2, 2, 2, 2], 'hard_integer'],
    # 'l1ratio_3': ['linspace', 0, 1, [2, 2, 2, 2, 2, 2], 'fixed_float'],
    # 'l1ratio_4': ['linspace', 0, 1, [2, 2, 2, 2, 2, 2], 'fixed_integer'],
    'max_depth': [[4, 5, 6], [3, 3, 3, 1, 1, 1], 'soft_integer'],
}

string_params = {
    'solver': [['lbfgs', 'saga'], 2]
}

# test_cls = DaskAutoGridSearch(
#                                 estimator,
#                                 numerical_params=numerical_params,
#                                 string_params=string_params,
#                                 total_passes=6,
#                                 total_passes_is_hard=True,
#                                 max_shifts=3,
#
#                                 # scoring=self.scoring,
#                                 # iid=self.iid,
#                                 # refit=self.refit,
#                                 # cv=self.cv,
#                                 # # verbose=0, (sklearn only)
#                                 # # pre_dispatch='2*n_jobs', (sklearn only)
#                                 # error_score=self.error_score,
#                                 # return_train_score=self.return_train_score,
#                                 # scheduler=self.scheduler,
#                                 # n_jobs=self.n_jobs,
#                                 # cache_cv=self.cache_cv
# )

test_cls = SKLearnAutoGridSearch(
    estimator,
    numerical_params=numerical_params,
    string_params=string_params,
    total_passes=6,
    total_passes_is_hard=True,
    max_shifts=3
)

test_cls.demo()

# END TEST demo() ########################################################################################################

quit()

estimator = 'dummy_estimator'
numerical_params = {'a': ['linspace', 0, 10, 11, 'soft_float']}
string_params = {'b': [['test1', 'test2']]}
test_cls = AutoGridSearch(
    estimator=estimator,
    numerical_params=numerical_params,
    string_params=string_params,
    total_passes=2,
    total_passes_is_hard=True,
    max_shifts=2,
)
# test_cls.__init__('dummy_estimator', numerical_params, string_params, 2)
# total_passes CAN ONLY BE AN INTEGER > 0 #########################################################################
for junk_value in [-1, 0, 'test', 3.1415]:
    try:
        test_cls.__init__(estimator, numerical_params, string_params, junk_value)
        raise TypeError
    except TypeError:
        raise Exception(f'\033[91mtotal_passes DID NOT EXCEPT FOR A NON-(INTEGER > 0): FAIL\033[0m')
    except Exception:
        print(f'\033[92mtotal_passes EXCEPTS IF NOT INTEGER > 0 ({junk_value}): PASS\033[0m')

for good_value in [3, 100]:
    try:
        test_cls.numerical_params['a'][3] = [good_value for _ in range(test_cls.total_passes)]
        test_cls.__init__(estimator, numerical_params, string_params, good_value)
        print(f'\033[92mtotal_passes DID NOT EXCEPT FOR AN INTEGER > 0: PASS\033[0m')
    except:
        raise Exception(f'\033[91mtotal_passes EXCEPTS FOR INTEGER > 0 ({good_value}): FAIL\033[0m')
# END total_passes DOES NOT EXCEPT FOR INTEGER > 0 #####################################################################
