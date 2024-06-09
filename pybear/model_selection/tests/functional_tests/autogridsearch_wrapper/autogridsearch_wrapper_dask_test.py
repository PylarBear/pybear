# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This tests
# module handles fit() for all dask gridsearch modules.



import pytest


pytest.skip(f"test takes 25 minutes", allow_module_level=True)

import numpy as np

from pybear.model_selection import autogridsearch_wrapper

from dask_ml.model_selection import (
    GridSearchCV as DaskGridSearchCV,
    RandomizedSearchCV as DaskRandomizedSearchCV,
    IncrementalSearchCV,
    HyperbandSearchCV,
    SuccessiveHalvingSearchCV,
    InverseDecaySearchCV
)


# this estimator needs to have partial_fit() for Incremental
from dask_ml.linear_model import LogisticRegression as dask_logistic




@pytest.fixture
def _dask_estimator_1():
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
def _dask_params_1():
    return {
        'C': [np.logspace(-5, 5, 3), [3,3,3], 'soft_float'],
        'solver': [['lbfgs', 'admm'], 2, 'string'],
        'fit_intercept': [[True, False], 2, 'bool']
    }



from dask_ml.datasets import make_classification as dask_make


@pytest.fixture
def _X_y():
    return dask_make(n_features=5, n_samples=100, chunks=(100,5), n_classes=2)


from distributed import Client




# dask gscvs that dont need a partial_fit exposed ** * ** * ** * ** * **


class TestDaskGSCVSThatDontNeedPartialFit:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.fixture(scope="module")
    def dask_client(self):
        with Client() as client:
            yield client


    @pytest.mark.parametrize('DASK_GSCV, _refit',
        (
            (DaskGridSearchCV, True),
            (DaskGridSearchCV, False),
            (DaskRandomizedSearchCV, True),
            (DaskRandomizedSearchCV, False),
        )
    )
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    def test_dask_gscvs(self, DASK_GSCV, _refit, _dask_estimator_1,
            _dask_params_1, _total_passes, _tpih, _max_shifts, _X_y, dask_client):

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
            _dask_estimator_1,
            _dask_params_1,
            total_passes=_total_passes,
            total_passes_is_hard=_tpih,
            max_shifts=_max_shifts,
            agscv_verbose=False,
            **_gscv_kwargs
        )

        del _gscv_kwargs

        assert _test_cls.total_passes >= len(_dask_params_1['C'][1])
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

            assert _test_cls.total_passes >= len(_dask_params_1['C'][1])
            assert _test_cls.total_passes_is_hard is _tpih
            assert _test_cls.max_shifts == _max_shifts
            assert _test_cls.agscv_verbose is False

            assert 'BALANCED_ACCURACY' in str(_test_cls.scorer_).upper()
            # cannot tests MockEstimator for scoring or scorer_

            assert isinstance(_test_cls.best_estimator_, type(_dask_estimator_1))
            best_params_ = _test_cls.best_params_
            assert isinstance(best_params_, dict)
            assert sorted(list(best_params_)) == sorted(list(_dask_params_1))
            assert all(map(
                isinstance,
                best_params_.values(),
                ((int, float, bool, str) for _ in _dask_params_1)
            ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# END dask gscvs that dont need a partial_fit exposed ** * ** * ** * ** *





# dask gscvs that need a partial_fit exposed ** * ** * ** * ** * ** * ** *


from sklearn.linear_model import SGDClassifier


@pytest.fixture
def _dask_estimator_2():
    return SGDClassifier(
        loss='hinge',
        penalty='l2',
        # alpha=0.0001,
        l1_ratio=0.15,
        # fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        # learning_rate='optimal',
        eta0=0.1,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False
    )



@pytest.fixture
def _dask_params_2():
    return {
        'alpha': [np.logspace(-5, 5, 3), [3,3,3], 'soft_float'],
        'learning_rate': [['constant', 'optimal'], 2, 'string'],
        'fit_intercept': [[True, False], 2, 'bool']
    }



class TestDaskGSCVSThatNeedPartialFitButNotSuccessiveHalving:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.fixture(scope="module")
    def dask_client(self):
        with Client() as client:
            yield client


    @pytest.mark.parametrize('DASK_GSCV',
         (IncrementalSearchCV, HyperbandSearchCV, InverseDecaySearchCV)
    )
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    def test_dask_gscvs(self, DASK_GSCV, _dask_estimator_2, _dask_params_2,
                    _total_passes, _tpih, _max_shifts, _X_y, dask_client):

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)

        _gscv_kwargs = {'scoring': 'balanced_accuracy'}

        _test_cls = AutoGridSearch(
            _dask_estimator_2,
            _dask_params_2,
            total_passes=_total_passes,
            total_passes_is_hard=_tpih,
            max_shifts=_max_shifts,
            agscv_verbose=False,
            **_gscv_kwargs
        )

        del _gscv_kwargs

        assert _test_cls.total_passes >= len(_dask_params_2['alpha'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False


        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        _test_cls.fit(*_X_y, classes=(0,1))

        assert _test_cls.total_passes >= len(_dask_params_2['alpha'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False

        assert 'BALANCED_ACCURACY' in str(_test_cls.scorer_).upper()
        # cannot tests MockEstimator for scoring or scorer_

        assert isinstance(_test_cls.best_estimator_, type(_dask_estimator_2))
        best_params_ = _test_cls.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_dask_params_2))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _dask_params_2)
        ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **






class TestDaskSuccessiveHalving:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.fixture(scope="module")
    def dask_client(self):
        with Client() as client:
            yield client


    @pytest.mark.parametrize('DASK_GSCV', (SuccessiveHalvingSearchCV, ))
    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    def test_dask_gscvs(self, DASK_GSCV, _dask_estimator_2, _dask_params_2,
                    _total_passes, _tpih, _max_shifts, _X_y, dask_client):

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)

        _gscv_kwargs = {'scoring': 'balanced_accuracy', 'n_initial_iter': 2}

        _test_cls = AutoGridSearch(
            _dask_estimator_2,
            _dask_params_2,
            total_passes=_total_passes,
            total_passes_is_hard=_tpih,
            max_shifts=_max_shifts,
            agscv_verbose=False,
            **_gscv_kwargs
        )

        del _gscv_kwargs

        assert _test_cls.total_passes >= len(_dask_params_2['alpha'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False


        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        _test_cls.fit(*_X_y, classes=(0, 1))

        assert _test_cls.total_passes >= len(_dask_params_2['alpha'][1])
        assert _test_cls.total_passes_is_hard is _tpih
        assert _test_cls.max_shifts == _max_shifts
        assert _test_cls.agscv_verbose is False

        assert 'BALANCED_ACCURACY' in str(_test_cls.scorer_).upper()
        # cannot tests MockEstimator for scoring or scorer_

        assert isinstance(_test_cls.best_estimator_, type(_dask_estimator_2))
        best_params_ = _test_cls.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_dask_params_2))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _dask_params_2)
        ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **






# END dask gscvs that need a partial_fit exposed ** * ** * ** * ** * ** *






























