# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This test
# module handles fit() for all dask gridsearch modules.



import pytest
import numpy as np

from pybear.model_selection import autogridsearch_wrapper

from pybear.model_selection.tests.functional_tests.autogridsearch_wrapper. \
    mock_estimator_test_fixture import MockEstimator



from dask_ml.model_selection import (
    GridSearchCV as DaskGridSearchCV,
    RandomizedSearchCV as DaskRandomizedSearchCV,
    IncrementalSearchCV,
    HyperbandSearchCV,
    SuccessiveHalvingSearchCV,
    InverseDecaySearchCV
)

# pizza -- this estimator needs to have partial_fit() for Incremental
from dask_ml.linear_model import LogisticRegression as dask_logistic

@pytest.fixture
def _dask_estimator():
    return dask_logistic(
        penalty='l2',
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        random_state=None,
        solver='admm',
        max_iter=100,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=1,
        solver_kwargs=None
    )

    # return MockEstimator()



@pytest.fixture
def _dask_params():
    return {
    'C': [np.logspace(-5, 5, 3), [3,3,3], 'soft_float'],
    'solver': [['lbfgs', 'admm'], 2, 'string']
}
#     return {
#         'param_a': [np.logspace(-5, 5, 3), [3, 3, 3], 'soft_float'],
#         'param_b': [[1, 2, 3], [3, 3, 3], 'fixed_integer']
#     }

from dask_ml.datasets import make_classification as dask_make


@pytest.fixture
def _X_y():
    return dask_make(n_features=5, n_samples=100, chunks=(100,5))


from distributed import Client






class TestDaskGSCVS:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.fixture(scope="module")
    def dask_client(self):   # must be self, not @staticmethod or pytest wont collect
        with Client() as client:
            yield client


    @pytest.mark.parametrize('DASK_GSCV, _refit',
        (
            (DaskGridSearchCV, True),
            (DaskGridSearchCV, False),
            (DaskRandomizedSearchCV, True),
            (DaskRandomizedSearchCV, False),
            (IncrementalSearchCV, 'na'),
            (HyperbandSearchCV, 'na'),
            (SuccessiveHalvingSearchCV, 'na'),
            (InverseDecaySearchCV, 'na')
        )
    )
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    def test_dask_gscvs(self, DASK_GSCV, _refit, _dask_estimator,
            _dask_params, _total_passes, _tpih, _max_shifts, _X_y, dask_client):

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)

        # 'refit' kwarg must be True for DaskGSCV and Randomized,
        # but Incremental, Hyperband, SuccessiveHalving, InverseDecay,
        # dont accept refit kwarg. Create a kwargs dict that manages what
        # kwargs are passed to whom.
        _has_refit_kwarg = \
            DASK_GSCV.__name__ in ['DaskGridSearchCV', 'DaskRandomizedSearchCV']
        _gscv_kwargs = {'scoring': 'balanced_accuracy'}
        if _has_refit_kwarg:
            _gscv_kwargs = _gscv_kwargs | {'refit': _refit, 'n_jobs': -1}
        else:
            pass   # do not put 'refit' in these


        _test_cls = AutoGridSearch(
            _dask_estimator,
            _dask_params,
            total_passes=_total_passes,
            total_passes_is_hard=_tpih,
            max_shifts=_max_shifts,
            agscv_verbose=False,
            **_gscv_kwargs
        )

        del _gscv_kwargs

        assert _test_cls.total_passes >= len(_dask_params['C'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False


        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        if _has_refit_kwarg and not _refit:

            assert _test_cls.refit is False

            with pytest.raises(AttributeError):
                _test_cls.fit(*_X_y)

        else:  # if has refit kwarg and refit=True or does not have refit kwarg

            _test_cls.fit(*_X_y)

            if _has_refit_kwarg:
                assert _test_cls.refit is True

            assert _test_cls.total_passes >= len(_dask_params['C'][1])
            assert _test_cls.total_passes_is_hard is _tpih
            assert _test_cls.max_shifts == _max_shifts
            assert _test_cls.agscv_verbose is False

            assert 'BALANCED_ACCURACY' in str(_test_cls.scorer_).upper()
            # cannot test MockEstimator for scoring or scorer_

            assert isinstance(_test_cls.best_estimator_, type(_dask_estimator))
            best_params_ = _test_cls.best_params_
            assert isinstance(best_params_, dict)
            assert sorted(list(best_params_)) == sorted(list(_dask_params))
            assert all(map(
                isinstance,
                best_params_.values(),
                ((int, float, bool, str) for _ in _dask_params)
            ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **




























