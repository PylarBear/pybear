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

from dask_ml.datasets import make_classification as dask_make

from dask_ml.linear_model import LogisticRegression as dask_logistic
# this estimator needs to have partial_fit() for Incremental
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection import (
    GridSearchCV as DaskGridSearchCV,
    RandomizedSearchCV as DaskRandomizedSearchCV,
    IncrementalSearchCV,
    HyperbandSearchCV,
    SuccessiveHalvingSearchCV,
    InverseDecaySearchCV
)

from distributed import Client




@pytest.fixture
def dask_client():
    client = Client(n_workers=1, threads_per_worker=1, asynchronous=False)
    yield client
    client.close()



@pytest.fixture
def _X_y():
    return dask_make(n_features=5, n_samples=100, chunks=(100,5), n_classes=2)


# ** * ** * ** * ** * ** ** * ** * ** * ** * ** ** * ** * ** * ** * **
# dask gscvs that dont need a partial_fit exposed ** * ** * ** * ** * **


class TestDaskGSCVSThatDontNeedPartialFit:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @staticmethod
    @pytest.fixture
    def _dask_estimator_1():

        return dask_logistic(
            penalty='l2',
            dual=False,
            tol=0.0001,
            C=1e-5,
            fit_intercept=False,
            intercept_scaling=1.0,
            class_weight=None,
            random_state=None,
            solver='newton',
            max_iter=100,
            multi_class='ovr',
            verbose=0,
            warm_start=False,
            n_jobs=None,
            solver_kwargs=None
        )

    @staticmethod
    @pytest.fixture
    def _dask_params_1():
        return {
            'C': [np.logspace(-5, 5, 3), 3, 'soft_float'],
            'solver': [['lbfgs', 'admm'], 2, 'fixed_string']
        }



    @pytest.mark.parametrize('DASK_GSCV',
        (DaskGridSearchCV, DaskRandomizedSearchCV)
    )
    @pytest.mark.parametrize('_total_passes', (2, ))
    @pytest.mark.parametrize('_scorer',
        ('accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (1, ))
    @pytest.mark.parametrize('_refit', ('accuracy', False, lambda x: 0))
    def test_dask_gscvs(self, _dask_estimator_1, _dask_params_1, DASK_GSCV,
        _total_passes, _scorer, _tpih, _max_shifts, _refit, _X_y
    ):

        # faster without client


        AGSCV_params = {
            'estimator': _dask_estimator_1,
            'params': _dask_params_1,
            'total_passes': _total_passes,
            'total_passes_is_hard': _tpih,
            'max_shifts': _max_shifts,
            'agscv_verbose': False,
            'scoring': _scorer,
            'n_jobs': -1,
            'cv': 4,
            'error_score': 'raise',
            'return_train_score': False,
            'refit': _refit,
            'iid': True,
            'cache_cv': True,
            'scheduler': None
        }

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)(**AGSCV_params)

        # 25_04_19 changed fit() to raise ValueError when best_params_
        # is not exposed. it used to be that agscv code was shrink-wrapped
        # around sklearn & dask_ml gscv quirks as to when they do/dont expose
        # best_params_. there are no longer any bandaids that condition params
        # for the parent gscvs to get them to "properly" expose 'best_params_',
        # and there are no more predictive shrink-wraps to block failure.
        # The user is left to die by however the parent gscv handles the exact
        # params as given. what that means here is that we are not going to
        # coddle to every little nuanced thing that makes a gscv not want to
        # expose 'best_params_'. Try to fit, if ValueError is raised, look to
        # see that 'best_params_' is not exposed and go to the next test.
        try:
            AutoGridSearch.fit(*_X_y)
            assert isinstance(getattr(AutoGridSearch, 'best_params_'), dict)
        except ValueError:
            assert not hasattr(AutoGridSearch, 'best_params_')
            pytest.skip(reason=f'cant do any later tests without fit')
        except Exception as e:
            raise e

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert AutoGridSearch.total_passes >= len(_dask_params_1['C'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        assert AutoGridSearch.scoring == _scorer
        assert AutoGridSearch.refit == _refit


        if _refit:
            assert isinstance(
                AutoGridSearch.best_estimator_, type(_dask_estimator_1)
            )
        elif not _refit:
            with pytest.raises(AttributeError):
                AutoGridSearch.best_estimator_


        best_params_ = AutoGridSearch.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_dask_params_1))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _dask_params_1)
        ))

        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# END dask gscvs that dont need a partial_fit exposed ** * ** * ** * ** *
# ** * ** * ** * ** * ** ** * ** * ** * ** * ** ** * ** * ** * ** * **



# ** * ** * ** * ** * ** ** * ** * ** * ** * ** ** * ** * ** * ** * **
# dask gscvs that need a partial_fit exposed ** * ** * ** * ** * ** * ** *


@pytest.fixture
def _dask_estimator_2():
    # has partial_fit method
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
        'alpha': [np.logspace(-5, 5, 3), [3,3], 'soft_float'],
        'learning_rate': [['constant', 'optimal'], 2, 'fixed_string'],
    }


# RuntimeError: Attempting to use an asynchronous Client in a synchronous context of `dask.compute`
@pytest.mark.skip(reason=f"as of 25_04_18 fails for async client. does anybody care.")
class TestDaskGSCVSThatNeedPartialFitButNotSuccessiveHalving:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.mark.parametrize('DASK_GSCV',
         (IncrementalSearchCV, HyperbandSearchCV, InverseDecaySearchCV)
    )
    @pytest.mark.parametrize('_total_passes', (2, ))
    @pytest.mark.parametrize('_scorer', (['accuracy'], ))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (1, ))
    def test_dask_gscvs(self, DASK_GSCV, _dask_estimator_2, _dask_params_2,
        _total_passes, _scorer, _tpih, _max_shifts, _X_y, dask_client
    ):

        # cannot accept multiple scorers
        # THIS ONE NEEDS A CLIENT

        AGSCV_params = {
            'estimator': _dask_estimator_2,
            'params': _dask_params_2,
            'total_passes': _total_passes,
            'total_passes_is_hard': _tpih,
            'max_shifts': _max_shifts,
            'agscv_verbose': False,
            'scoring': _scorer,
            'verbose': False,
            'random_state': None,
            'tol': 0.0001,
            'prefix': '',
            'test_size': None,
            'patience': False,
            'max_iter': 100,
            # 'n_initial_parameters': 10,
            # 'decay_rate': 10,
            # 'n_initial_iter': None,
            # 'fits_per_score': 1,
            # 'aggressiveness': 3,
            # 'predict_meta': None,
            # 'predict_proba_meta': None,
            # 'transform_meta': None,
            # 'scores_per_fit': None
        }

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)(**AGSCV_params)

        AutoGridSearch.fit(*_X_y, classes=(0, 1))

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert AutoGridSearch.total_passes >= len(_dask_params_2['alpha'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        if isinstance(_scorer, list) and len(_scorer) == 1:
            assert AutoGridSearch.scoring == _scorer[0]
        else:
            assert AutoGridSearch.scoring == _scorer


        assert isinstance(AutoGridSearch.best_estimator_, type(_dask_estimator_2))

        best_params_ = AutoGridSearch.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_dask_params_2))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _dask_params_2)
        ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# RuntimeError: Attempting to use an asynchronous Client in a synchronous context of `dask.compute`
@pytest.mark.skip(reason=f"as of 25_04_18 fails for async client. does anybody care.")
class TestDaskSuccessiveHalving:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.mark.parametrize('DASK_GSCV', (SuccessiveHalvingSearchCV, ))
    @pytest.mark.parametrize('_total_passes', (2, ))
    @pytest.mark.parametrize('_scorer', (['accuracy'], ))
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (1, ))
    def test_dask_gscvs(self, DASK_GSCV, _dask_estimator_2, _dask_params_2,
        _total_passes, _scorer, _tpih, _max_shifts, _X_y, dask_client
    ):

        # cannot accept multiple scorers
        # THIS ONE NEEDS A CLIENT

        AGSCV_params = {
            'estimator': _dask_estimator_2,
            'params': _dask_params_2,
            'total_passes': _total_passes,
            'total_passes_is_hard': _tpih,
            'max_shifts': _max_shifts,
            'agscv_verbose': False,
            'scoring': _scorer,
            'verbose': False,
            'random_state': None,
            'tol': 1e-3,
            'prefix': '',
            'test_size': None,
            'patience': False,
            'max_iter': None,
            'n_initial_parameters': 10,
            'n_initial_iter': 3,
            'aggressiveness': 3
        }

        AutoGridSearch = autogridsearch_wrapper(DASK_GSCV)(**AGSCV_params)

        AutoGridSearch.fit(*_X_y, classes=(0, 1))

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert AutoGridSearch.total_passes >= len(_dask_params_2['alpha'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        if isinstance(_scorer, list) and len(_scorer) == 1:
            assert AutoGridSearch.scoring == _scorer[0]
        else:
            assert AutoGridSearch.scoring == _scorer

        assert isinstance(AutoGridSearch.best_estimator_, type(_dask_estimator_2))

        best_params_ = AutoGridSearch.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_dask_params_2))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str) for _ in _dask_params_2)
        ))


        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **



# END dask gscvs that need a partial_fit exposed ** * ** * ** * ** * ** *
# ** * ** * ** * ** * ** ** * ** * ** * ** * ** ** * ** * ** * ** * **








