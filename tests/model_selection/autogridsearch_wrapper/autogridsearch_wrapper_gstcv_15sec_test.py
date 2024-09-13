# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# this module tests compatibility of autogridsearch_wrapper with GSTCV
# simply by running wrapped GSTCV to completion and asserting a few of
# the GSTCV attributes are exposed by the wrapper.



# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This test
# module handles fit() for the GSTCV module.



import pytest
import numpy as np

from sklearn.datasets import make_classification as sk_make

from pybear.model_selection import autogridsearch_wrapper

from pybear.model_selection import GSTCV




class TestGSTCV:

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
        return {
            'param_a': [np.logspace(-5, -3, 3), [3, 3, 3], 'soft_float'],
            'param_b': [[1, 2], [2, 2, 2], 'fixed_integer']
        }


    @staticmethod
    @pytest.fixture
    def _X_y():
        return sk_make(n_features=5, n_samples=100)

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_total_passes', (2, 3, 4))
    @pytest.mark.parametrize('_scorer',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_tpih', (True, ))
    @pytest.mark.parametrize('_max_shifts', (2, ))
    @pytest.mark.parametrize('_refit', ('accuracy', False, lambda x: 0))
    def test_GSTCV(self, mock_estimator_test_fixture, _params, _total_passes,
        _scorer, _tpih, _max_shifts, _refit, _X_y
    ):

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
            'refit': _refit
        }

        # autogridsearch_wrapper __init__ does validation... should
        # raise for blocked conditions (when a gscv parent would not
        # expose best_params_)
        if len(_scorer) > 1 and _refit is False:
            with pytest.raises(AttributeError):
                autogridsearch_wrapper(GSTCV)(**AGSTCV_params)
            pytest.skip(reason=f'cant do any later tests without init')
        else:
            AutoGridSearch = autogridsearch_wrapper(GSTCV)(**AGSTCV_params)


        AutoGridSearch.fit(*_X_y)

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        assert AutoGridSearch.total_passes >= len(_params['param_a'][1])
        assert AutoGridSearch.total_passes_is_hard is _tpih
        assert AutoGridSearch.max_shifts == _max_shifts
        assert AutoGridSearch.agscv_verbose is False
        if isinstance(_scorer, list) and len(_scorer)==1:
            assert AutoGridSearch.scoring == _scorer[0]
        else:
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






























