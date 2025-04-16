# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _estimator import _estimator

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import LogisticRegression as sk_Logistic

from dask_ml.linear_model import LogisticRegression as dask_Logistic



# 24_06_01 currently does not test for presence of fit(), get_params(),
# and set_params()



class TestEstimator:

    @pytest.mark.parametrize('non_class',
    (0, 1, 3.14, [1,2], (1,2), {1,2}, {'a':1}, 'junk', lambda x: x)
    )
    def test_rejects_anything_not_a_class(self, non_class):
        with pytest.raises(TypeError):
            _estimator({'a': [[1,2,3], [3,3,3], 'fixed_integer']}, non_class)


    def test_accepts_estimators(self):

        good_params = {'penalty': [['l1', 'l2'], 2, 'string']}

        _estimator(good_params, SGDClassifier())
        _estimator(good_params, SGDRegressor())
        _estimator(good_params, sk_Logistic())



@pytest.fixture
def bad_params():
    return {
        'xqv': [[10, 100, 1000], [3,11,11], 'soft_float'],
        'zmp': [[3, 4, 5], [3, 3, 3], 'fixed_integer'],
        '@*&!': [['a', 'b', 'c'], 3, 'string']
    }



class TestSKLearnAttrs:

    def test_rejects_bad_attrs(self, bad_params):
        with pytest.raises(AttributeError):
            _estimator(bad_params, sk_Logistic())

    def test_accepts_good_attrs(self):
        good_params = {
            'penalty': [['l1', 'l2', 'elasticnet'], 3, 'string'],
            'C': [np.logspace(-5, 5, 6), [6, 11, 11], 'soft_float'],
            'fit_intercept': [[True, False], 3, 'fixed_float'],
            'solver': [['lbfgs', 'saga'], 2, 'string'],
            'max_iter': [[10000, 15000, 20000], [3, 11, 11], 'soft_integer'],
            'n_jobs': [[1,2,3,4], 2, 'fixed_integer'],
            'l1_ratio': [np.linspace(0, 1, 11), [11, 6, 6], 'hard_float']
        }

        _estimator(good_params, sk_Logistic())




class TestDaskAttrs:

    def test_rejects_bad_attrs(self, bad_params):
        with pytest.raises(AttributeError):
            _estimator(bad_params, dask_Logistic())

    def test_accepts_good_attrs(self):
        good_params = {
            'penalty': [['l1', 'l2', 'elasticnet'], 3, 'string'],
            'C': [np.logspace(-5, 5, 6), [6, 11, 11], 'soft_float'],
            'fit_intercept': [[True, False], 3, 'fixed_float'],
            'solver': [['lbfgs', 'saga'], 2, 'string'],
            'max_iter': [[10000, 15000, 20000], [3, 11, 11], 'soft_integer'],
            'n_jobs': [[1,2,3,4], 2, 'fixed_integer'],
            'intercept_scaling': [np.linspace(0, 1, 11), [11, 6, 6], 'hard_float']
        }

        _estimator(good_params, dask_Logistic())







