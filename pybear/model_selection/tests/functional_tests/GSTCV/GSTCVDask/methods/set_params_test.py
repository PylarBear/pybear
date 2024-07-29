# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from dask_ml.linear_model import (
    LogisticRegression as dask_Logistic,
    LinearRegression as dask_LinearRegression
)
from xgboost.dask import DaskXGBClassifier

from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from dask_ml.feature_extraction.text import CountVectorizer as dask_CountVectorizer
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask




class TestDaskSetParams:

    # not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture
    def _estimator():
        return dask_Logistic(
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
            solver_kwargs=None,
        )


    @staticmethod
    @pytest.fixture
    def _param_grid():
        return {'C': np.logspace(-5, -2, 4)}


    @staticmethod
    @pytest.fixture
    def _GSTCVDask(_estimator, _param_grid):
        return GSTCVDask(
            _estimator,
            _param_grid,
            thresholds = None,
            scoring = 'accuracy',
            iid = True,
            refit = True,
            cv = None,
            verbose = 0,
            error_score = 'raise',
            return_train_score = False,
            scheduler = None,
            n_jobs = None,
            cache_cv = False
        )

    # END not pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** *

    # pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @staticmethod
    @pytest.fixture
    def _pipeline():
        return Pipeline(
            steps=[
                ('dask_onehot', dask_OneHotEncoder()),
                ('dask_logistic', dask_Logistic())
            ]
        )


    @staticmethod
    @pytest.fixture
    def _pipe_param_grid():
        return {
            'dask_onehot__drop': [None, 'first'],
            'dask_logistic__C': np.logspace(-5, -2, 4)
        }


    @staticmethod
    @pytest.fixture
    def _GSTCVDask_PIPE(_pipeline, _pipe_param_grid):
        return GSTCVDask(
            _pipeline,
            _pipe_param_grid,
            thresholds = None,
            scoring = 'accuracy',
            iid = True,
            refit = True,
            cv = None,
            verbose = 0,
            error_score = 'raise',
            return_train_score = False,
            scheduler = None,
            n_jobs = None,
            cache_cv = False
        )

    # END pipeline fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** *



    @pytest.mark.parametrize('junk_param',
        (0, 1, 3.14, None, True, 'trash', [0,1], (0, 1), min, lambda x: x)
    )
    def test_rejects_junk_params(self, junk_param, _GSTCVDask, _GSTCVDask_PIPE):
        with pytest.raises(TypeError):
            _GSTCVDask.set_params(junk_param)

        with pytest.raises(TypeError):
            _GSTCVDask_PIPE.set_params(junk_param)



    def test_rejects_invalid_params(self, _GSTCVDask, _GSTCVDask_PIPE):

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

    def test_accepts_good_params_shallow_no_pipe(self, _GSTCVDask):

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


        good_params_shallow = _GSTCVDask.get_params(deep=False)

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


    def test_accepts_good_params_deep_no_pipe(self, _GSTCVDask):

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


        good_params_deep_no_pipe = _GSTCVDask.get_params(deep=True)

        good_params_deep_no_pipe['estimator__tol'] = 1e-6
        good_params_deep_no_pipe['estimator__C'] = 1e-3
        good_params_deep_no_pipe['estimator__fit_intercept'] = False
        good_params_deep_no_pipe['estimator__solver'] = 'saga'
        good_params_deep_no_pipe['estimator__max_iter'] = 10_000
        good_params_deep_no_pipe['estimator__n_jobs'] = 8

        _GSTCVDask.set_params(**good_params_deep_no_pipe)

        assert _GSTCVDask.estimator.tol == 1e-6
        assert _GSTCVDask.estimator.C == 1e-3
        assert _GSTCVDask.estimator.fit_intercept is False
        assert _GSTCVDask.estimator.solver == 'saga'
        assert _GSTCVDask.estimator.max_iter == 10_000
        assert _GSTCVDask.estimator.n_jobs == 8

        # END deep no pipe ** *** ** *** ** *** ** *** ** *** ** *** **


    def test_accepts_good_params_shallow_pipe(self, _GSTCVDask_PIPE):

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

        good_params_pipe_shallow = _GSTCVDask_PIPE.get_params(deep=False)

        good_params_pipe_shallow['estimator'] = \
            Pipeline(steps=[('bag_of_words', dask_CountVectorizer()),
                            ('xgboost', DaskXGBClassifier())])
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
        assert isinstance(_GSTCVDask_PIPE.estimator.steps[0][1], dask_CountVectorizer)
        assert isinstance(_GSTCVDask_PIPE.estimator.steps[1][1], DaskXGBClassifier)
        assert _GSTCVDask_PIPE.param_grid == \
               {'bag_of_words__analyzer': ['word', 'char', 'char_wb'],
                'xgboost__max_depth': [3, 4, 5]}
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.n_jobs == 4
        assert _GSTCVDask_PIPE.cv == 5
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True

        # END shallow pipe ** *** ** *** ** *** ** *** ** *** ** *** **


    def test_accepts_good_params_deep_pipe(self, _GSTCVDask_PIPE):

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

        good_params_pipe_deep = _GSTCVDask_PIPE.get_params(deep=True)

        good_params_pipe_deep['cv'] = 12
        good_params_pipe_deep['estimator__dask_onehot__drop'] = 'first'
        good_params_pipe_deep['estimator__dask_logistic__C'] = 1e-3
        good_params_pipe_deep['estimator__dask_logistic__max_iter'] = 10000
        good_params_pipe_deep['estimator__dask_logistic__n_jobs'] = None
        good_params_pipe_deep['n_jobs'] = 8
        good_params_pipe_deep['param_grid'] = {
            'dask_onehot__drop': [None, 'first'],
            'dask_logistic__C': [0.0001, 0.001, 0.01]}
        good_params_pipe_deep['refit'] = False
        good_params_pipe_deep['return_train_score'] = True
        good_params_pipe_deep['scoring'] = 'balanced_accuracy'
        good_params_pipe_deep['verbose'] = 10

        _GSTCVDask_PIPE.set_params(**good_params_pipe_deep)

        assert _GSTCVDask_PIPE.cv == 12
        assert _GSTCVDask_PIPE.estimator.steps[0][0] == 'dask_onehot'
        assert _GSTCVDask_PIPE.estimator.steps[0][1].drop == 'first'
        assert _GSTCVDask_PIPE.estimator.steps[1][0] == 'dask_logistic'
        assert _GSTCVDask_PIPE.estimator.steps[1][1].C == 1e-3
        assert _GSTCVDask_PIPE.estimator.steps[1][1].max_iter == 10000
        assert _GSTCVDask_PIPE.estimator.steps[1][1].n_jobs == None
        assert _GSTCVDask_PIPE.n_jobs == 8
        assert _GSTCVDask_PIPE.param_grid == {
            'dask_onehot__drop': [None, 'first'],
            'dask_logistic__C': [0.0001, 0.001, 0.01]
        }
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.verbose == 10

        # END deep pipe ** *** ** *** ** *** ** *** ** *** ** *** ** ***












































