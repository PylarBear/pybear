# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# tests if fixed integer and fixed float can take negative numbers

# 24_06_01 AutoGridSearch conclusively cannot take negative numbers for
# any data/search type. Already knew that soft and hard cant because of
# the universal hard bounds for soft_float, soft_int, hard_float, and
# hard_int. But fixed cannot take negative numbers either because of all
# the validation that uses np.log10... cant do log on negative numbers.

# tests if fixed integer can take zeros
# the code does accept fixed integer of zero, without error. accuracy of
# result was not tested.


from model_selection import autogridsearch_wrapper
import numpy as np
from typing import Union

from sklearn.model_selection import GridSearchCV as skl_GridSearchCV
from sklearn.datasets import make_regression, make_classification
from pybear.model_selection.tests.functional_tests.autogridsearch_wrapper. \
    mock_estimator_test_fixture import MockEstimator


est = MockEstimator()


AutoGridSearch = autogridsearch_wrapper(skl_GridSearchCV)


agscv = AutoGridSearch(
    estimator=est,
    params={'param_a': [[4, 10, 20], [3, 3, 3], 'soft_float'],
            'param_b': [[0, 200, 300], [3, 3, 3], 'fixed_integer']
    },
    agscv_verbose=True
)


X, y = make_classification(n_samples=100, n_features=5)

agscv.fit(X, y)

















