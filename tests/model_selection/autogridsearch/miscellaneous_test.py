# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import autogridsearch_wrapper

from pybear.model_selection import GSTCV, GSTCVDask

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from sklearn.linear_model import LogisticRegression as sk_Logistic



class TestMiscellaneousStuff:


    @staticmethod
    @pytest.fixture(scope='module')
    def _make_agscv():

        def foo(_gscv):

            _agscv = autogridsearch_wrapper(_gscv)

            return _agscv

        return foo

    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # this should raise ValueError for not exposing best_params_
    def test_best_params_not_exposed(self, _make_agscv):

        _agscv = _make_agscv(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False,
            scoring=['accuracy', 'balanced_accuracy']
        )

        with pytest.raises(ValueError):
            _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    # pizza this should go under _param_conditioner
    def test_converts_scorer_list_of_one_to_str(self, _make_agscv):

        _agscv = _make_agscv(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False,
            scoring=['accuracy']
        )

        # pizza come back and finalize this once we know how the conditioned params are stored
        # assert _agscv.scoring == 'accuracy'


    # pizza think on whether this is needed
    @pytest.mark.parametrize('_refit', (True, False, ['accuracy'], lambda x: 0))
    def test_sklearn_GSCV_indifferent_to_refit(self, _make_agscv, _refit):

        _agscv = _make_agscv(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={'solver': [['saga', 'lbfgs'], 2, 'string']},
            refit=_refit
        )

        _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    # pizza u need to fix this then u may want to find a better home for this
    @pytest.mark.skip(f"25_04_19 pizza needs a fix!")
    def test_accepts_empty_params(self, _make_agscv):

        _agscv = _make_agscv(sk_GridSearchCV)(
            estimator=sk_Logistic(),
            params={},
            refit=False
        )

        _agscv.fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))



    # GSTCV(Dask) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('gscv_parent', (GSTCV, GSTCVDask))
    def test_accepts_threshold_kwarg(self, _make_agscv, gscv_parent):

        _thresholds = np.linspace(0, 1, 3)

        _agscv = _make_agscv(gscv_parent)(
            estimator=sk_Logistic(),
            params={},
            refit=False,
            thresholds=_thresholds
        )

        assert np.array_equiv(_agscv.thresholds, _thresholds)


    # END GSTCV(Dask) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **












