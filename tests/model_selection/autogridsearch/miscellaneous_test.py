# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from sklearn.linear_model import LogisticRegression as sk_Logistic



class TestMiscellaneousStuff:


    # this should raise ValueError for not exposing best_params_
    def test_best_params_not_exposed(self):

        _agscv = autogridsearch_wrapper(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False,
            scoring=['accuracy', 'balanced_accuracy']
        )

        with pytest.raises(ValueError):
            _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    # pizza think on whether this is needed
    @pytest.mark.parametrize('_refit', (True, False, ['accuracy'], lambda x: 0))
    def test_sklearn_GSCV_indifferent_to_refit(self, _refit):

        _agscv = autogridsearch_wrapper(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={'solver': [['saga', 'lbfgs'], 2, 'string']},
            refit=_refit
        )

        _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    # pizza u need to fix this then u may want to find a better home for this
    @pytest.mark.skip(f"25_04_19 pizza needs a fix!")
    def test_accepts_empty_params(self):

        _agscv = autogridsearch_wrapper(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False
        )

        _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))














