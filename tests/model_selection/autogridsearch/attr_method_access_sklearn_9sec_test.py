# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression as sk_logistic
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from pybear.base._is_fitted import is_fitted


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES
class Fixtures:


    @staticmethod
    @pytest.fixture(scope='module')
    def _total_passes():
        return 3


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs(_total_passes):
        return {
            'estimator': sk_logistic(),
            'params': {
                'C': [np.logspace(-2, 2, 5), 5, 'soft_float'],
                'fit_intercept': [[True, False], 2, 'fixed_bool']
            },
            'total_passes': _total_passes,
            'total_passes_is_hard': True,
            'max_shifts': 2,
            'agscv_verbose': False
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _attrs():
        return [
            'best_score_'
            'best_params_',
            'GRIDS_',
            'RESULTS_'
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _methods():
        return [
            'fit',
            'get_params',
            '_agscv_reset',
            'set_params'
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100,5)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_shape):
        return np.random.uniform(0, 1, _shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _y(_shape):
        return np.random.randint(0, 2, (_shape[0],))


    @staticmethod
    @pytest.fixture(scope='module')
    def _agscv():
        return autogridsearch_wrapper(sk_GridSearchCV)



# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



# ACCESS ATTR BEFORE AND AFTER FIT
class TestAttrAccessBeforeAndAfterFit(Fixtures):


    def test_attr_access_before_fit(
        self, _X, _y, _agscv, _kwargs, _total_passes, _attrs
    ):

        TestCls = _agscv(**_kwargs)

        # BEFORE FIT ***************************************************

        # SHOULD GIVE AttributeError
        for attr in _attrs:
            with pytest.raises(AttributeError):
                getattr(TestCls, attr)

        # END BEFORE FIT ***********************************************


    def test_attr_access_after_fit(
        self, _X, _y, _agscv, _kwargs, _total_passes, _attrs
    ):

        TestCls = _agscv(**_kwargs)

        # AFTER FIT ****************************************************

        TestCls.fit(_X, _y)

        # 'best_score_',
        # 'best_params_',
        # 'GRIDS_',
        # 'RESULTS_'

        # after fit, should have access to everything

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _best_score = getattr(TestCls, 'best_score_')
        assert isinstance(_best_score, numbers.Real)
        # setting of best_score_ is controlled by parent GSCV
        # would rather that this could not be set
        setattr(TestCls, 'best_score_', 1)
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _best_params = getattr(TestCls, 'best_params_')
        assert isinstance(_best_params, dict)
        assert len(_best_params) == 2
        assert isinstance(_best_params['C'], numbers.Real)
        assert isinstance(_best_params['fit_intercept'], bool)
        # setting of best_params_ is controlled by parent GSCV
        # would rather that this could not be set
        setattr(TestCls, 'best_params_', {})
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _GRIDS = getattr(TestCls, 'GRIDS_')
        assert isinstance(_GRIDS, dict)
        assert all(map(
            isinstance,
            _GRIDS.keys(),
            (numbers.Integral for _ in _GRIDS.keys())
        ))
        assert len(_GRIDS) == _total_passes
        assert isinstance(_GRIDS[_total_passes - 1]['C'], list)
        assert isinstance(_GRIDS[_total_passes - 1]['C'][0], numbers.Real)
        assert isinstance(_GRIDS[_total_passes - 1]['fit_intercept'], list)
        assert isinstance(_GRIDS[_total_passes - 1]['fit_intercept'][0], bool)
        # GRIDS_ cannot be set
        with pytest.raises(AttributeError):
            setattr(TestCls, 'GRIDS_', {})
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _RESULTS = getattr(TestCls, 'RESULTS_')
        assert isinstance(_RESULTS, dict)
        assert all(map(
            isinstance,
            _RESULTS.keys(),
            (numbers.Integral for _ in _RESULTS.keys())
        ))
        assert len(_RESULTS) == _total_passes
        assert isinstance(_RESULTS[_total_passes - 1]['C'], numbers.Real)
        assert isinstance(_RESULTS[_total_passes - 1]['fit_intercept'], bool)
        # RESULTS_ cannot be set
        with pytest.raises(AttributeError):
            setattr(TestCls, 'RESULTS_', {})
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # END AFTER FIT ************************************************

        del TestCls

# END ACCESS ATTR BEFORE AND AFTER FIT


# ACCESS METHODS BEFORE AND AFTER FIT ***
class TestMethodAccessBeforeAndAfterFit(Fixtures):


    def test_access_methods_before_fit(self, _X, _y, _agscv, _kwargs, _attrs):

        TestCls = _agscv(**_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # fit()
        assert not is_fitted(TestCls)
        assert isinstance(TestCls.fit(_X, _y), _agscv)
        assert is_fitted(TestCls)

        # get_params()
        out = TestCls.get_params(True)
        assert isinstance(out, dict)
        # the wrapper
        assert 'agscv_verbose' in out
        assert isinstance(out['agscv_verbose'], bool)
        # the parent
        assert 'n_jobs' in out
        assert isinstance(out['n_jobs'], (numbers.Integral, type(None)))

        # set_params()
        # the wrapper
        assert isinstance(TestCls.set_params(agscv_verbose=True), _agscv)
        assert TestCls.agscv_verbose is True
        assert isinstance(TestCls.set_params(agscv_verbose=False), _agscv)
        assert TestCls.agscv_verbose is False
        # the parent
        assert isinstance(TestCls.set_params(scoring='balanced_accuracy'), _agscv)
        assert TestCls.scoring == 'balanced_accuracy'
        assert isinstance(TestCls.set_params(scoring='accuracy'), _agscv)
        assert TestCls.scoring == 'accuracy'

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************


    def test_access_methods_after_fit(self, _X, _y, _agscv, _kwargs):

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        FittedTestCls = _agscv(**_kwargs)

        # fit()
        assert isinstance(FittedTestCls.fit(_X, _y), _agscv)

        assert hasattr(FittedTestCls, 'GRIDS_')
        assert hasattr(FittedTestCls, 'RESULTS_')

        # get_params()
        out = FittedTestCls.get_params(True)
        # the wrapper
        assert 'total_passes_is_hard' in out
        assert isinstance(out['total_passes_is_hard'], bool)
        # the parent
        assert 'n_jobs' in out
        assert isinstance(out['n_jobs'], (numbers.Integral, type(None)))

        # set_params()
        # the wrapper
        assert isinstance(FittedTestCls.set_params(agscv_verbose=True), _agscv)
        assert FittedTestCls.agscv_verbose is True
        assert isinstance(FittedTestCls.set_params(agscv_verbose=False), _agscv)
        assert FittedTestCls.agscv_verbose is False
        # the parent
        assert isinstance(FittedTestCls.set_params(refit=True), _agscv)
        assert FittedTestCls.refit is True
        assert isinstance(FittedTestCls.set_params(refit=None), _agscv)
        assert FittedTestCls.refit is None

        del FittedTestCls

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT








