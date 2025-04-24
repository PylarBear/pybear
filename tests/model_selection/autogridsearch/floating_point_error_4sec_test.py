# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection import autogridsearch_wrapper
from sklearn.model_selection import GridSearchCV as sk_GSCV
from sklearn.linear_model import Ridge as sk_RR
from sklearn.datasets import make_regression

import pytest



class TestFloatingPointError:


    def test_floating_point_error(self):

        AGSCV = autogridsearch_wrapper(sk_GSCV)
        _params = {
            'alpha': [[1e-300, 1e-299, 1e-199], 3, 'soft_float'],
            'fit_intercept': [[True, False], [2, 1, 1], 'fixed_bool']
        }
        agscv = AGSCV(estimator=sk_RR(), params=_params, total_passes=3)
        X, y = make_regression(n_samples=20, n_features=2, n_informative=2)

        with pytest.raises(ValueError):
            agscv.fit(X, y)







