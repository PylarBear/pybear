# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from copy import deepcopy
import dask.array as da

from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression
)

from xgboost import XGBClassifier as sk_XGBClassifier
import distributed

from dask_ml.preprocessing import StandardScaler as dask_StandardScaler
from dask_ml.feature_extraction.text import CountVectorizer as dask_CountVectorizer
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask



# 24_08_05 GSTCV fixtures have been scoped to class level to avoid
# fitting repeatedly when fixture is test level. What this has caused is
# that the params need to be reset to the original state after some tests.


@pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0), scope='class')
class TestDaskSetParams:


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
        return distributed.Client(n_workers=None, threads_per_worker=1)


    # not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='class')
    def _estimator():
        return sk_XGBClassifier(
            max_depth=5,
            n_estimators=50,
            tree_method='hist'
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _param_grid():
        return {'max_depth': [3,4,5]}


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCVDask_prefit(_estimator, _param_grid, _refit):

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
            n_jobs=None,
            cache_cv=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCVDask_postfit(_GSTCVDask_prefit, _client, X, y):

        return _GSTCVDask_prefit.fit(X, y)

    # END not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** *

    # pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @staticmethod
    @pytest.fixture(scope='class')
    def _pipeline(_estimator):
        return Pipeline(
            steps=[
                ('dask_standardscaler', dask_StandardScaler()),
                ('sk_xgb', _estimator)
            ]
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _pipe_param_grid():
        return {
            'dask_standardscaler__with_mean': [True],
            'sk_xgb__max_depth': [4]
        }


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCVDask_PIPE_prefit(_pipeline, _pipe_param_grid, _refit):

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
            n_jobs=None,
            cache_cv=False
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _GSTCVDask_PIPE_postfit(_GSTCVDask_PIPE_prefit, _client, X, y):

        return _GSTCVDask_PIPE_prefit.fit(X, y)

    # END pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('junk_param',
        (0, 1, 3.14, None, True, 'trash', [0,1], (0, 1), min, lambda x: x)
    )
    def test_rejects_junk_params(
            self, junk_param, state, _GSTCVDask_prefit, _GSTCVDask_PIPE_prefit,
            _GSTCVDask_postfit, _GSTCVDask_PIPE_postfit,
    ):

        if state == 'prefit':
            _GSTCVDask = _GSTCVDask_prefit
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_prefit
        elif state == 'postfit':
            _GSTCVDask = _GSTCVDask_postfit
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_postfit

        with pytest.raises(TypeError):
            _GSTCVDask.set_params(junk_param)

        with pytest.raises(TypeError):
            _GSTCVDask_PIPE.set_params(junk_param)


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_rejects_invalid_params(
        self, state, _GSTCVDask_prefit, _GSTCVDask_PIPE_prefit, _GSTCVDask_postfit,
        _GSTCVDask_PIPE_postfit
    ):

        if state == 'prefit':
            _GSTCVDask = _GSTCVDask_prefit
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_prefit
        elif state == 'postfit':
            _GSTCVDask = _GSTCVDask_postfit
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_postfit

        # just check param names
        # invalid values for params should be caught at fit() by _validate()
        bad_params = dask_LinearRegression().get_params(deep=True)

        with pytest.raises(ValueError):
            _GSTCVDask.set_params(**bad_params)

        with pytest.raises(ValueError):
            _GSTCVDask_PIPE.set_params(**bad_params)


    # for shallow and deep no pipe / pipe, just take all the params from
    # itself and verify accepts everything; change some of the params and
    # assert new settings are correct

    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_accepts_good_params_shallow_no_pipe(
        self, state, _GSTCVDask_prefit, _GSTCVDask_postfit
    ):

        # shallow no pipe ** *** ** *** ** *** ** *** ** *** ** *** **
        # GSTCV(Dask) shallow params no pipe (FROM SKLEARN NOT DASK!):
        # 'estimator': ...(),
        # 'param_grid': {...},
        # 'thresholds': None,
        # 'scoring': 'accuracy',
        # 'n_jobs': None,
        # 'pre_dispatch': '2*n_jobs',
        # 'cv': None,
        # 'refit': True,
        # 'verbose': 0,
        # 'error_score': 'raise',
        # 'return_train_score': False

        if state == 'prefit':
            _GSTCVDask = _GSTCVDask_prefit
        elif state == 'postfit':
            _GSTCVDask = _GSTCVDask_postfit

        good_params_shallow = _GSTCVDask.get_params(deep=False)
        original_good_params_shallow = deepcopy(good_params_shallow)

        good_params_shallow['thresholds'] = [0.1, 0.5, 0.9]
        good_params_shallow['scoring'] = 'balanced_accuracy'
        good_params_shallow['n_jobs'] = 4
        good_params_shallow['cv'] = 8
        good_params_shallow['refit'] = False
        good_params_shallow['verbose'] = 10
        good_params_shallow['return_train_score'] = True

        _GSTCVDask.set_params(**good_params_shallow)

        assert _GSTCVDask.thresholds == [0.1, 0.5, 0.9]
        assert _GSTCVDask.scoring == 'balanced_accuracy'
        assert _GSTCVDask.n_jobs == 4
        assert _GSTCVDask.cv == 8
        assert _GSTCVDask.refit is False
        assert _GSTCVDask.verbose == 10
        assert _GSTCVDask.return_train_score is True

        # END shallow no pipe ** *** ** *** ** *** ** *** ** *** ** ***

        # now that GSTCV no pipe fixture is scoped to class, reset the
        # params to the original before going to the next test
        _GSTCVDask.set_params(**original_good_params_shallow)

        del original_good_params_shallow


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_accepts_good_params_deep_no_pipe(
        self, state, _GSTCVDask_prefit, _GSTCVDask_postfit
    ):

        # deep no pipe ** *** ** *** ** *** ** *** ** *** ** *** ** ***
        # GSTCV(Dask) deep params -- no pipe (FROM SKLEARN NOT DASK!):
        # 'estimator': LogisticRegression(),
        # 'param_grid': {'C': [0.0001, 0.001, 0.01]},
        # 'thresholds': None,
        # 'scoring': 'accuracy',
        # 'n_jobs': None,
        # 'pre_dispatch': '2*n_jobs',
        # 'cv': None,
        # 'refit': True,
        # 'verbose': 0,
        # 'error_score': 'raise',
        # 'return_train_score': False,
        # 'estimator__penalty': 'l2',
        # 'estimator__dual': False,
        # 'estimator__tol': 0.0001,
        # 'estimator__C': 1.0,
        # 'estimator__fit_intercept': True,
        # 'estimator__intercept_scaling': 1,
        # 'estimator__class_weight': None,
        # 'estimator__random_state': None,
        # 'estimator__solver': 'lbfgs',
        # 'estimator__max_iter': 100,
        # 'estimator__multi_class': 'deprecated',
        # 'estimator__verbose': 0,
        # 'estimator__warm_start': False,
        # 'estimator__n_jobs': None,
        # 'estimator__l1_ratio': None

        if state == 'prefit':
            _GSTCVDask = _GSTCVDask_prefit
        elif state == 'postfit':
            _GSTCVDask = _GSTCVDask_postfit

        good_params_deep_no_pipe = _GSTCVDask.get_params(deep=True)

        good_params_deep_no_pipe['estimator__max_depth'] = 6
        good_params_deep_no_pipe['estimator__n_estimators'] = 300
        good_params_deep_no_pipe['estimator__tree_method'] = 'exact'

        _GSTCVDask.set_params(**good_params_deep_no_pipe)

        assert _GSTCVDask.estimator.max_depth == 6
        assert _GSTCVDask.estimator.n_estimators == 300
        assert _GSTCVDask.estimator.tree_method == 'exact'

        # END deep no pipe ** *** ** *** ** *** ** *** ** *** ** *** **


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_accepts_good_params_shallow_pipe(
        self, state, _GSTCVDask_PIPE_prefit, _GSTCVDask_PIPE_postfit
    ):

        # shallow pipe ** *** ** *** ** *** ** *** ** *** ** *** ** ***
        # GSTCV(Dask) shallow params -- pipe (DROM SKLEARN NOT DASK!):
        # 'estimator': Pipeline(steps=[('onehot', OneHotEncoder()),
        #                       ('logistic', LogisticRegression())]),
        # 'param_grid': {'onehot__min_frequency': [3, 4, 5],
        #                   'logistic__C': [0.0001, 0.001, 0.01]},
        # 'thresholds': None,
        # 'scoring': 'accuracy',
        # 'n_jobs': None,
        # 'pre_dispatch': '2*n_jobs',
        # 'cv': None,
        # 'refit': True,
        # 'verbose': 0,
        # 'error_score': 'raise',
        # 'return_train_score': False

        if state == 'prefit':
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_prefit
        elif state == 'postfit':
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_postfit

        good_params_pipe_shallow = _GSTCVDask_PIPE.get_params(deep=False)
        original_good_params_pipe_shallow = deepcopy(good_params_pipe_shallow)

        good_params_pipe_shallow['estimator'] = \
            Pipeline(steps=[('bag_of_words', dask_CountVectorizer()),
                            ('xgboost', sk_XGBClassifier())])
        good_params_pipe_shallow['param_grid'] = \
            {'bag_of_words__analyzer': ['word', 'char', 'char_wb'],
             'xgboost__max_depth': [3, 4, 5]}
        good_params_pipe_shallow['scoring'] = 'balanced_accuracy'
        good_params_pipe_shallow['n_jobs'] = 4
        good_params_pipe_shallow['cv'] = 5
        good_params_pipe_shallow['refit'] = False
        good_params_pipe_shallow['return_train_score'] = True

        _GSTCVDask_PIPE.set_params(**good_params_pipe_shallow)

        assert isinstance(_GSTCVDask_PIPE.estimator, Pipeline)
        assert isinstance(_GSTCVDask_PIPE.estimator.steps[0][1],
                          dask_CountVectorizer)
        assert isinstance(_GSTCVDask_PIPE.estimator.steps[1][1],
                          sk_XGBClassifier)
        assert _GSTCVDask_PIPE.param_grid == \
               {'bag_of_words__analyzer': ['word', 'char', 'char_wb'],
                'xgboost__max_depth': [3, 4, 5]}
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.n_jobs == 4
        assert _GSTCVDask_PIPE.cv == 5
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True

        # END shallow pipe ** *** ** *** ** *** ** *** ** *** ** *** **

        # now that GSTCV no pipe fixture is scoped to class, reset the
        # params to the original before going to the next test
        _GSTCVDask_PIPE.set_params(**original_good_params_pipe_shallow)

        del original_good_params_pipe_shallow


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_accepts_good_params_deep_pipe(
        self, state, _GSTCVDask_PIPE_prefit, _GSTCVDask_PIPE_postfit
    ):

        # deep pipe ** *** ** *** ** *** ** *** ** *** ** *** ** *** **
        # GSTCV(Dask) deep params -- pipe (FROM SKLEARN NOT DASK!):
        # 'cv': None,
        # 'error_score': nan,
        # 'estimator__memory': None,
        # 'estimator__steps': [('onehot', OneHotEncoder()),
        #                       ('logistic', LogisticRegression())],
        # 'estimator__verbose': False,
        # 'estimator__sk_onehot': OneHotEncoder(),
        # 'estimator__sk_logistic': LogisticRegression(),
        # 'estimator__sk_onehot__categories': 'auto',
        # 'estimator__sk_onehot__drop': None,
        # 'estimator__sk_onehot__dtype': <class 'numpy.float64'>,
        # 'estimator__sk_onehot__feature_name_combiner': 'concat',
        # 'estimator__sk_onehot__handle_unknown': 'error',
        # 'estimator__sk_onehot__max_categories': None,
        # 'estimator__sk_onehot__min_frequency': None,
        # 'estimator__sk_onehot__sparse_output': True,
        # 'estimator__sk_logistic__C': 1.0,
        # 'estimator__sk_logistic__class_weight': None,
        # 'estimator__sk_logistic__dual': False,
        # 'estimator__sk_logistic__fit_intercept': True,
        # 'estimator__sk_logistic__intercept_scaling': 1,
        # 'estimator__sk_logistic__l1_ratio': None,
        # 'estimator__sk_logistic__max_iter': 100,
        # 'estimator__sk_logistic__multi_class': 'deprecated',
        # 'estimator__sk_logistic__n_jobs': None,
        # 'estimator__sk_logistic__penalty': 'l2',
        # 'estimator__sk_logistic__random_state': None,
        # 'estimator__sk_logistic__solver': 'lbfgs',
        # 'estimator__sk_logistic__tol': 0.0001,
        # 'estimator__sk_logistic__verbose': 0,
        # 'estimator__sk_logistic__warm_start': False,
        # 'estimator': Pipeline(steps=[('sk_onehot', OneHotEncoder()),
        #                 ('sk_logistic', LogisticRegression())]),
        # 'n_jobs': None,
        # 'param_grid': {'sk_onehot__min_frequency': [3, 4, 5],
        #               'sk_logistic__C': [0.0001, 0.001, 0.01]},
        # 'pre_dispatch': '2*n_jobs',
        # 'refit': True,
        # 'return_train_score': False,
        # 'scoring': None,
        # 'verbose': 0

        if state == 'prefit':
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_prefit
        elif state == 'postfit':
            _GSTCVDask_PIPE = _GSTCVDask_PIPE_postfit

        good_params_pipe_deep = _GSTCVDask_PIPE.get_params(deep=True)

        good_params_pipe_deep['cv'] = 12
        good_params_pipe_deep['estimator__dask_standardscaler__with_mean'] = False
        good_params_pipe_deep['estimator__sk_xgb__max_depth'] = 6
        good_params_pipe_deep['estimator__sk_xgb__n_estimators'] = 300
        good_params_pipe_deep['estimator__sk_xgb__tree_method'] = 'exact'
        good_params_pipe_deep['n_jobs'] = 8
        good_params_pipe_deep['param_grid'] = {
            'dask_standardscaler__with_mean': [True, False],
            'sk_xgb__max_depth': [3, 4, 5]}
        good_params_pipe_deep['refit'] = False
        good_params_pipe_deep['return_train_score'] = True
        good_params_pipe_deep['scoring'] = 'balanced_accuracy'
        good_params_pipe_deep['verbose'] = 10

        _GSTCVDask_PIPE.set_params(**good_params_pipe_deep)

        assert _GSTCVDask_PIPE.cv == 12
        assert _GSTCVDask_PIPE.estimator.steps[0][0] == 'dask_standardscaler'
        assert _GSTCVDask_PIPE.estimator.steps[0][1].with_mean == False
        assert _GSTCVDask_PIPE.estimator.steps[1][0] == 'sk_xgb'
        assert _GSTCVDask_PIPE.estimator.steps[1][1].max_depth == 6
        assert _GSTCVDask_PIPE.estimator.steps[1][1].n_estimators == 300
        assert _GSTCVDask_PIPE.estimator.steps[1][1].tree_method == 'exact'
        assert _GSTCVDask_PIPE.n_jobs == 8
        assert _GSTCVDask_PIPE.param_grid == {
            'dask_standardscaler__with_mean': [True, False],
            'sk_xgb__max_depth': [3, 4, 5]
        }
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.verbose == 10

        # END deep pipe ** *** ** *** ** *** ** *** ** *** ** *** ** ***












































