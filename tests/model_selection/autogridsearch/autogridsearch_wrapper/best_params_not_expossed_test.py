# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from sklearn.linear_model import LogisticRegression as sk_Logistic



class TestBestParamsNotExposed:


    # this should raise ValueError for not exposing best_params_
    def test_best_params_not_exposed(self):

        _agscv = autogridsearch_wrapper(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False,
            scoring=['accuracy', 'balanced_accuracy']
        )

        with pytest.raises(ValueError):
            _agscv.fit(
                np.random.uniform(0,1,(20,10)),
                np.random.randint(0,2,(20,))
            )






