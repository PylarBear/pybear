# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This tests
# module handles fit() for all sklearn gridsearch modules.



import pytest

import numpy as np

from pybear.model_selection import autogridsearch_wrapper

from sklearn.datasets import make_classification as sk_make

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV as SklearnGridSearchCV,
    RandomizedSearchCV as SklearnRandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)


class TestSklearnGSCVS:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def _estimator():
        return sk_LogisticRegression()


    @staticmethod
    @pytest.fixture
    def _params():
        return {
            'C': [np.logspace(-5, 5, 3), [3, 3, 3], 'soft_float'],
            'fit_intercept': [[True, False], 2, 'bool']
        }


    @staticmethod
    @pytest.fixture
    def _X_y():
        return sk_make(n_features=5, n_samples=100)

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('SKLEARN_GSCV',
        (
            SklearnGridSearchCV,
            SklearnRandomizedSearchCV,
            HalvingGridSearchCV,
            HalvingRandomSearchCV
        )
    )
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_scorer',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    @pytest.mark.parametrize('_refit', ('accuracy', False, lambda x: 0))
    def test_sklearn_gscvs(self, _estimator, _params, SKLEARN_GSCV,
        _total_passes, _scorer, _tpih, _max_shifts, _refit, _X_y
    ):

        # the 'halving' grid searches cannot take multiple scorers
        if SKLEARN_GSCV in (HalvingGridSearchCV, HalvingRandomSearchCV) \
                and len(_scorer) > 1:
            pytest.skip(
                reason=f"the 'halving' grid searches cannot take multiple scorers"
            )

        AGSCV_params = {
            'estimator': _estimator,
            'params': _params,
            'total_passes': _total_passes,
            'total_passes_is_hard': _tpih,
            'max_shifts': _max_shifts,
            'agscv_verbose': False,
            'scoring': _scorer,
            'n_jobs': -1,
            'cv': 4,
            'verbose': 0,
            'error_score': 'raise',
            'return_train_score': False,
            'refit': _refit
        }

        # autogridsearch_wrapper __init__ does validation... should
        # raise for blocked conditions (when a gscv parent would not
        # expose best_params_)
        if len(_scorer) > 1 and _refit is False:
            with pytest.raises(AttributeError):
                autogridsearch_wrapper(SKLEARN_GSCV)(**AGSCV_params)
            pytest.skip(reason=f'cant do any later tests without init')
        else:
            AutoGridSearch = autogridsearch_wrapper(SKLEARN_GSCV)(**AGSCV_params)


        AutoGridSearch.fit(*_X_y)

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert AutoGridSearch.total_passes >= len(_params['C'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        if isinstance(_scorer, list) and len(_scorer)==1:
            assert AutoGridSearch.scoring == _scorer[0]
        else:
            assert AutoGridSearch.scoring == _scorer
        assert AutoGridSearch.refit == _refit

        if _refit:
            assert isinstance(AutoGridSearch.best_estimator_, type(_estimator))
        elif not _refit:
            with pytest.raises(AttributeError):
                AutoGridSearch.best_estimator_


        best_params_ = AutoGridSearch.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_params))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _params)
        ))

        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **





















