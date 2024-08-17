# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np


from sklearn.linear_model import LogisticRegression as sk_Logistic



from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV


# Tests GSTCV get_params against sk_GSCV get_params for shallow & deep
# 24_08_05 GSTCV fitted fixture has been scoped to class level to avoid
# fitting repeatedly when fixture is test level. What this has caused is
# that the params need to be reset to the original state after some tests.


@pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0), scope='class')
class TestSKGetParams:

    @staticmethod
    @pytest.fixture(scope='class')
    def X():
        return np.random.randint(0, 10, (1000, 5))


    @staticmethod
    @pytest.fixture(scope='class')
    def y():
        return np.random.randint(0, 2, (1000,))








    # not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='class')
    def _estimator():
        return sk_Logistic(
            tol=1e-6,
            C=13,
            solver="lbfgs",
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _param_grid():
        return {'C': [1e-5], 'tol': [1e-6]}


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_prefit(_estimator, _param_grid, _refit):

        return sk_GridSearchCV(
            _estimator,
            _param_grid,
            scoring='accuracy',
            n_jobs=-1,
            refit=_refit,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score='raise',
            return_train_score=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_postfit(_GSCV_prefit, X, y):

        return _GSCV_prefit.fit(X, y)


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_prefit(_estimator, _param_grid, _refit):

        return GSTCV(
            _estimator,
            _param_grid,
            thresholds=None,
            scoring='accuracy',
            n_jobs=-1,
            refit=_refit,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score='raise',
            return_train_score=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_postfit(_GSTCV_prefit, X, y):

        return _GSTCV_prefit.fit(X, y)
    # END not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** *

    # pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @staticmethod
    @pytest.fixture(scope='class')
    def _pipeline(_estimator):
        return Pipeline(
            steps=[
                ('sk_onehot', sk_OneHotEncoder(sparse_output=False)),
                ('sk_logistic', _estimator)
            ]
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _pipe_param_grid():
        return {
            'sk_onehot__min_frequency': [5],
            'sk_onehot__max_categories': [3],
            'sk_logistic__C': [1e-5],
            'sk_logistic__tol': [1e-6]
        }


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_PIPE_prefit(_pipeline, _pipe_param_grid, _refit):

        return sk_GridSearchCV(
            _pipeline,
            _pipe_param_grid,
            scoring='accuracy',
            n_jobs=1,
            refit=_refit,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score='raise',
            return_train_score=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_PIPE_postfit(_GSCV_PIPE_prefit, X, y):

        return _GSCV_PIPE_prefit.fit(X, y)


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_PIPE_prefit(_pipeline, _pipe_param_grid, _refit):

        return GSTCV(
            _pipeline,
            _pipe_param_grid,
            thresholds=None,
            scoring='accuracy',
            n_jobs=-1,
            refit=_refit,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score='raise',
            return_train_score=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_PIPE_postfit(_GSTCV_PIPE_prefit, X, y):

        return _GSTCV_PIPE_prefit.fit(X, y)


    # END pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, state, bad_deep, _GSTCV_prefit, _GSTCV_postfit
    ):

        if state == 'prefit':
            _GSTCV = _GSTCV_prefit
        elif state == 'postfit':
            _GSTCV = _GSTCV_postfit

        with pytest.raises(ValueError):
            _GSTCV.get_params(bad_deep)


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_shallow_no_pipe(
        self, state, _GSCV_prefit, _GSCV_postfit, _GSTCV_prefit, _GSTCV_postfit
    ):

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit':
            _GSCV = _GSCV_prefit
            _GSTCV = _GSTCV_prefit
        elif state == 'postfit':
            _GSCV = _GSCV_postfit
            _GSTCV = _GSTCV_postfit

        # only test params' names, not values; GSTCV's defaults may be
        # different than skGSCV's

        skgscv_shallow = list(_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        # +1 for thresholds
        assert len(gstcv_shallow) == len(skgscv_shallow) + 1

        gstcv_shallow.remove('thresholds')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow no pipe ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_deep_no_pipe(
        self, state, _GSCV_prefit, _GSTCV_prefit, _GSCV_postfit, _GSTCV_postfit
    ):
        # test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit':
            _GSCV = _GSCV_prefit
            _GSTCV = _GSTCV_prefit
        elif state == 'postfit':
            _GSCV = _GSCV_postfit
            _GSTCV = _GSTCV_postfit

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        # +1 for thresholds
        assert len(gstcv_deep) == len(skgscv_deep) + 1

        gstcv_deep.remove('thresholds')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe_shallow(
        self, state, _GSCV_PIPE_prefit, _GSTCV_PIPE_prefit, _GSCV_PIPE_postfit,
        _GSTCV_PIPE_postfit
    ):

        # test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit':
            _GSCV_PIPE = _GSCV_PIPE_prefit
            _GSTCV_PIPE = _GSTCV_PIPE_prefit
        elif state == 'postfit':
            _GSCV_PIPE = _GSCV_PIPE_postfit
            _GSTCV_PIPE = _GSTCV_PIPE_postfit

        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_onehot'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_onehot'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_shallow = list(_GSCV_PIPE.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV_PIPE.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        # +1 for thresholds
        assert len(gstcv_shallow) == len(skgscv_shallow) + 1

        assert len(skgscv_shallow) == 10
        assert len(gstcv_shallow) == 11

        gstcv_shallow.remove('thresholds')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe_deep(
        self, state, _GSCV_PIPE_prefit, _GSTCV_PIPE_prefit, _GSCV_PIPE_postfit,
        _GSTCV_PIPE_postfit
    ):

        # test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        if state == 'prefit':
            _GSCV_PIPE = _GSCV_PIPE_prefit
            _GSTCV_PIPE = _GSTCV_PIPE_prefit
        elif state == 'postfit':
            _GSCV_PIPE = _GSCV_PIPE_postfit
            _GSTCV_PIPE = _GSTCV_PIPE_postfit

        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_onehot'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_onehot'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV_PIPE.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV_PIPE.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        # +1 for thresholds
        assert len(gstcv_deep) == len(skgscv_deep) + 1

        assert len(skgscv_deep) == 38
        assert len(gstcv_deep) == 39

        gstcv_deep.remove('thresholds')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *










