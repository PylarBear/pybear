# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza, all of the fixtures and code below is a direct, unchanged,
# copy from sklearn agscv test. modify as needed for gstcv


# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This test
# module handles fit() for the GridSearchThresholdCV module.



import pytest
import numpy as np

from pybear.model_selection import autogridsearch_wrapper

from pybear.model_selection.tests.functional_tests.autogridsearch_wrapper. \
    mock_estimator_test_fixture import MockEstimator







from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import (
    GridSearchCV as SklearnGridSearchCV,
    RandomizedSearchCV as SklearnRandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)



@pytest.fixture
def _sklearn_estimator():
    # using a mock estimator to significantly reduce fit() time
    return MockEstimator()


@pytest.fixture
def _sklearn_params():

    return {
        'param_a': [np.logspace(-5, 5, 3), [3, 3, 3], 'soft_float'],
        'param_b': [[1, 2, 3], [3, 3, 3], 'fixed_integer']
    }


from sklearn.datasets import make_classification as sk_make

@pytest.fixture
def _X_y():
    return sk_make(n_features=5, n_samples=100)


@pytest.xfail(reason=f"pizza not started, not done.")
class TestGSThresholdCV:

    # pizza come back and finish this when GSTCV is done

    assert 1 == 0   # make fail so xfail works

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.mark.parametrize('SKLEARN_GSCV',
        (
            SklearnGridSearchCV,
            SklearnRandomizedSearchCV,
            HalvingGridSearchCV,
            HalvingRandomSearchCV
        )
    )
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    @pytest.mark.parametrize('_refit', (True, False))
    def test_sklearn_gscvs(self, SKLEARN_GSCV, _sklearn_estimator,
       _sklearn_params, _total_passes, _tpih, _max_shifts, _refit, _X_y):

        AutoGridSearch = autogridsearch_wrapper(SKLEARN_GSCV)

        _test_cls = AutoGridSearch(
            _sklearn_estimator,
            _sklearn_params,
            total_passes=_total_passes,
            total_passes_is_hard=_tpih,
            max_shifts=_max_shifts,
            agscv_verbose=False,
            scoring='balanced_accuracy',
            refit=_refit,
            n_jobs=-1
        )

        _test_cls.fit(*_X_y)

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert _test_cls.total_passes >= len(_sklearn_params['param_a'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False

        # cannot test MockEstimator for scoring or scorer_

        if _refit:
            assert isinstance(_test_cls.best_estimator_, type(_sklearn_estimator))
        elif not _refit:
            with pytest.raises(AttributeError):
                _test_cls.best_estimator_


        best_params_ = _test_cls.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_sklearn_params))
        assert all(map(
                        isinstance,
                        best_params_.values(),
                        ((int, float, bool, str) for _ in _sklearn_params)
        ))
        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **






























