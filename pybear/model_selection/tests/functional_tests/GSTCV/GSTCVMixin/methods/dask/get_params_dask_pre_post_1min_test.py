# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import dask.array as da

from xgboost import XGBClassifier as sk_XGBClassifier
import distributed


from sklearn.preprocessing import StandardScaler as dask_StandardScaler
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


# Tests GSTCV get_params against dask_GSCV get_params for shallow & deep
# 24_08_05 GSTCV fitted fixture has been scoped to class level to avoid
# fitting repeatedly when fixture is test level. What this has caused is
# that the params need to be reset to the original state after some tests.


@pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0), scope='class')
class TestDaskGetParams:

    @staticmethod
    @pytest.fixture(scope='class')
    def X():
        return da.random.randint(0, 10, (1000, 5)).rechunk((1000, 5))


    @staticmethod
    @pytest.fixture(scope='class')
    def y():
        return da.random.randint(0, 2, (1000,)).rechunk((1000, ))


    @staticmethod
    @pytest.fixture(scope='class')
    def _client():
        yield distributed.Client(n_workers=None, threads_per_worker=1)


    # not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='class')
    def _estimator():
        return sk_XGBClassifier(
            n_estimators=50,
            max_depth=5,
            tree_method='hist'
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _param_grid():
        return {'max_depth': [5]}


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_prefit(_estimator, _param_grid, _refit):

        return dask_GridSearchCV(
            _estimator,
            _param_grid,
            scoring='accuracy',
            iid=True,
            refit=_refit,
            cv=None,
            error_score='raise',
            return_train_score=False,
            scheduler=None,
            n_jobs=-1,
            cache_cv=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_postfit(_GSCV_prefit, _client, X, y):

        return _GSCV_prefit.fit(X, y)


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_prefit(_estimator, _param_grid, _refit):

        return GSTCVDask(
            _estimator,
            _param_grid,
            thresholds=None,
            scoring='accuracy',
            iid=True,
            refit=_refit,
            cv=None,
            verbose=0,
            error_score='raise',
            return_train_score=False,
            scheduler=None,
            n_jobs=-1,
            cache_cv=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_postfit(_GSTCV_prefit, _client, X, y):

        return _GSTCV_prefit.fit(X, y)
    # END not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** *

    # pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @staticmethod
    @pytest.fixture(scope='class')
    def _pipeline(_estimator):
        return Pipeline(
            steps=[
                ('standardscaler', dask_StandardScaler()),
                ('logistic', _estimator)
            ]
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _pipe_param_grid():
        return {
            'standardscaler__with_mean': [True],
            'logistic__max_depth': [5],
            'logistic__n_estimators': [100]
        }


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_PIPE_prefit(_pipeline, _pipe_param_grid, _refit):

        return dask_GridSearchCV(
            _pipeline,
            _pipe_param_grid,
            scoring='accuracy',
            iid=True,
            refit=_refit,
            cv=None,
            error_score='raise',
            return_train_score=False,
            scheduler=None,
            n_jobs=-1,
            cache_cv=True
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSCV_PIPE_postfit(_GSCV_PIPE_prefit, _client, X, y):

        return _GSCV_PIPE_prefit.fit(X, y)


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_PIPE_prefit(_pipeline, _pipe_param_grid, _refit):

        return GSTCVDask(
            _pipeline,
            _pipe_param_grid,
            thresholds=None,
            scoring='accuracy',
            iid=True,
            refit=_refit,
            cv=None,
            verbose=0,
            error_score='raise',
            return_train_score=False,
            scheduler=None,
            n_jobs=-1,
            cache_cv=True
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCV_PIPE_postfit(_GSTCV_PIPE_prefit, _client, X, y):

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
        self, state, _GSCV_prefit, _GSTCV_prefit, _GSCV_postfit, _GSTCV_postfit
    ):

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit':
            _GSCV = _GSCV_prefit
            _GSTCV = _GSTCV_prefit
        elif state == 'postfit':
            _GSCV = _GSCV_postfit
            _GSTCV = _GSTCV_postfit

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +2 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')
        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_deep_no_pipe(self,
        state, _GSCV_prefit, _GSTCV_prefit, _GSCV_postfit, _GSTCV_postfit
    ):
        # test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit':
            _GSCV = _GSCV_prefit
            _GSTCV = _GSTCV_prefit
        elif state == 'postfit':
            _GSCV = _GSCV_postfit
            _GSTCV = _GSTCV_postfit

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_deep = list(_GSCV_postfit.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV_postfit.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep
        assert 'verbose' in gstcv_deep
        # +2 for thresholds / verbose
        assert len(gstcv_deep) == len(daskgscv_deep) + 2

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')
        assert np.array_equiv(daskgscv_deep, gstcv_deep)

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

        assert _GSCV_PIPE.estimator.steps[0][0] == 'standardscaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'standardscaler'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'logistic'

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(_GSCV_PIPE_postfit.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV_PIPE_postfit.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +1 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        assert len(daskgscv_shallow) ==11
        assert len(gstcv_shallow) == 13

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')
        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # END test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe_deep(
        self, state, _GSCV_PIPE_prefit, _GSTCV_PIPE_prefit, _GSCV_PIPE_postfit,
        _GSTCV_PIPE_postfit, _client
    ):

        # test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        if state == 'prefit':
            _GSCV_PIPE = _GSCV_PIPE_prefit
            _GSTCV_PIPE = _GSTCV_PIPE_prefit
        elif state == 'postfit':
            _GSCV_PIPE = _GSCV_PIPE_postfit
            _GSTCV_PIPE = _GSTCV_PIPE_postfit

        assert _GSCV_PIPE.estimator.steps[0][0] == 'standardscaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'standardscaler'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'logistic'

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_deep = list(_GSCV_PIPE.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV_PIPE.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep
        assert 'verbose' in gstcv_deep
        # +2 for thresholds / verbose
        assert len(gstcv_deep) == len(daskgscv_deep) + 2

        assert len(daskgscv_deep) == 58
        assert len(gstcv_deep) == 60

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')
        assert np.array_equiv(daskgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *










