# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this module tests compatibility of autogridsearch_wrapper with GSTCVDask
# simply by running wrapped GSTCVDask to completion and asserting a few of
# the GSTCVDask attributes are exposed by the wrapper.



# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This test
# module handles fit() for the GSTCVDask module.



import pytest
import numpy as np
from distributed import Client
from dask_ml.datasets import make_classification as dask_make

from pybear.model_selection import autogridsearch_wrapper

from pybear.model_selection import GSTCVDask


# pizza
pytest.skip(reason=f'5 minute test', allow_module_level=True)



class TestGSTCVDask:

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
    def _params():
        # using a mock estimator to significantly reduce fit() time
        return {
            'param_a': [np.logspace(-5, 5, 3), [3, 3], 'soft_float'],
            'param_b': [[1, 2], [2, 2], 'fixed_integer']
        }


    @staticmethod
    @pytest.fixture
    def _X_y():
        return dask_make(n_features=5, n_samples=100, chunks=(100, 5))


    @staticmethod
    @pytest.fixture(scope='module')
    def _client():
        client = Client(n_workers=None, threads_per_worker=1)
        yield client
        client.close()


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    def test_accepts_threshold_kwarg(self, mock_estimator_test_fixture):

        _thresholds = np.linspace(0, 1, 3)

        _agscv = autogridsearch_wrapper(GSTCVDask)(
            estimator=mock_estimator_test_fixture,
            params={},
            refit=False,
            thresholds=_thresholds
        )

        assert np.array_equiv(_agscv.thresholds, _thresholds)



    @pytest.mark.parametrize('_total_passes', (2, ))
    @pytest.mark.parametrize('_scorer',
        ('accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    @pytest.mark.parametrize('_refit', ('accuracy', False, lambda x: 0))
    def test_GSTCV(self,mock_estimator_test_fixture, _params, _total_passes,
        _scorer, _tpih, _max_shifts, _refit, _X_y, _client
    ):

        # faster with _client

        AGSTCV_params = {
            'estimator': mock_estimator_test_fixture,
            'params': _params,
            'total_passes': _total_passes,
            'total_passes_is_hard': _tpih,
            'max_shifts': _max_shifts,
            'agscv_verbose': False,
            'thresholds': [0.4, 0.6],
            'scoring': _scorer,
            'n_jobs': None,
            'cv': 4,
            'verbose': 0,
            'error_score': 'raise',
            'return_train_score': False,
            'refit': _refit,
            'iid': True,
            'cache_cv': True,
            'scheduler': None
        }

        AutoGridSearch = autogridsearch_wrapper(GSTCVDask)(**AGSTCV_params)

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
        assert AutoGridSearch.total_passes >= len(_params['param_a'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        assert AutoGridSearch.scoring == _scorer
        assert AutoGridSearch.refit == _refit

        # cannot test MockEstimator for scoring or scorer_

        if _refit:
            assert isinstance(
                AutoGridSearch.best_estimator_,
                type(mock_estimator_test_fixture)
            )
        else:
            with pytest.raises(AttributeError):
                AutoGridSearch.best_estimator_


        best_params_ = AutoGridSearch.best_params_
        assert isinstance(best_params_, dict)
        assert sorted(list(best_params_)) == sorted(list(_params))
        assert all(map(
            isinstance,
            best_params_.values(),
            ((int, float, bool, str, np.int64) for _ in _params)
        ))

        # best_threshold_ should always be exposed with one scorer
        if isinstance(_refit, str) or callable(_scorer) or \
                isinstance(_scorer, str) or len(_scorer) == 1:
            best_threshold_ = AutoGridSearch.best_threshold_
            assert isinstance(best_threshold_, float)
            assert 0 <= best_threshold_ <= 1

        # END assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * **






























