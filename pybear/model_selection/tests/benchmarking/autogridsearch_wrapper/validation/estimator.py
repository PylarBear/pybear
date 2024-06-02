# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
from sklearn.linear_model import LogisticRegression

from model_selection.autogridsearch._autogridsearch_wrapper._validation._estimator \
    import _estimator


sk_logistic = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='deprecated',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)





param_grid1 = {
    'a': [[1,2,3], [3,3,3], 'fixed_integer'],
    'b': [[10_000, 12_000, 14_000], [3,11,11], 'soft_integer'],
    'c': [np.logspace(-5,5,11), [11, 11, 11], 'soft_float']
}

param_grid2 = {
    'n_jobs': [[1,2,3], [3,3,3], 'fixed_integer'],
    'max_iter': [[10_000, 12_000, 14_000], [3,11,11], 'soft_integer'],
    'C': [np.logspace(-5,5,11), [11, 11, 11], 'soft_float']
}

for param in param_grid1:
    print(f'{param}: {hasattr(sk_logistic, param)} --- should be False')

for param in param_grid2:
    print(f'{param}: {hasattr(sk_logistic, param)} --- should be True')





try:
    _estimator(param_grid2, sk_logistic)
    print(f'_estimator correctly passed param_grid2')
except:
    print(f'_estimator wrongfully failed param_grid2')

try:
    _estimator(param_grid1, sk_logistic)
    print(f'_estimator wrongfully passed param_grid1')
except:
    print(f'_estimator correctly failed param_grid1')




