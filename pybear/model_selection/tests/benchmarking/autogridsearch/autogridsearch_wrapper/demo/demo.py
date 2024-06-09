# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegressionCV


AutoGridSearchCV = autogridsearch_wrapper(GridSearchCV)

params = {
    'Cs': ['logspace', 0, 4, 3, 'hard_integer'],
    'solver': [['lbfgs', 'saga'], 2, 'string'],
    'cv': [[3,4,5], [3,3,3], 'hard_integer']
}

estimator = LogisticRegressionCV



agscv_demo_class = AutoGridSearchCV(
    estimator,
    params,
    total_passes=3
)


agscv_demo_class.demo(true_best_params=None)
























