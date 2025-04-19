# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification



@pytest.fixture
def _estimator():
    return LogisticRegression(
        penalty='l2',
        dual=False,
        tol=0.0001,
        # C=1.0,
        # fit_intercept=True,
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




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


class TestBoolInFixedInteger:


    @pytest.fixture
    def _params(self):
        return {
            'C': [np.logspace(-4, 4, 3), [3, 3, 3], 'soft_float'],
            'fit_intercept': [[True, False], [2, 1, 1], 'fixed_integer']
        }


    def test_bool_in_fixed_integer(self, _estimator, _params):

        AutoGridSearch = autogridsearch_wrapper(GridSearchCV)

        X, y = make_classification(n_samples=50, n_features=5)

        with pytest.raises(TypeError):
            _test_cls = AutoGridSearch(
                _estimator,
                _params,
                total_passes=3,
                total_passes_is_hard=True,
                max_shifts=2,
                agscv_verbose=False
            )

            _test_cls.fit(X, y)




class TestBoolInFixedFloat:


    @pytest.fixture
    def _params(self):
        return {
            'C': [np.logspace(-4, 4, 3), [3, 3, 3], 'soft_float'],
            'fit_intercept': [[True, False], [2, 1, 1], 'fixed_float']
        }


    def test_bool_in_fixed_integer(self, _estimator, _params):

        AutoGridSearch = autogridsearch_wrapper(GridSearchCV)

        X, y = make_classification(n_samples=50, n_features=5)

        with pytest.raises(TypeError):
            _test_cls = AutoGridSearch(
                _estimator,
                _params,
                total_passes=3,
                total_passes_is_hard=True,
                max_shifts=2,
                agscv_verbose=False
            )

            _test_cls.fit(X, y)


















