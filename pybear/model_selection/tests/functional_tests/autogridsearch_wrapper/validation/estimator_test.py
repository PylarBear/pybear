# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
from model_selection.autogridsearch._autogridsearch_wrapper._validation._estimator \
    import _estimator

from sklearn.linear_model import SGDClassifier, SGDRegressor
from dask_ml.linear_model import LinearRegression, LogisticRegression


class TestEstimator:

    @pytest.mark.parametrize('non_class',
    (0, 1, 3.14, [1,2], (1,2), {1,2}, {'a':1}, 'junk', lambda x: x)
    )
    def test_rejects_anything_not_a_class(self, non_class):
        with pytest.raises(TypeError):
            _estimator(non_class)


    def test_accepts_estimators(self):
        _estimator(SGDClassifier)
        _estimator(SGDRegressor)
        _estimator(LinearRegression)
        _estimator(LogisticRegression)


