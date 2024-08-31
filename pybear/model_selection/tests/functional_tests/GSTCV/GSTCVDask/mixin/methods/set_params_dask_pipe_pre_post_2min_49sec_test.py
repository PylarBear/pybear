# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from copy import deepcopy

from dask_ml.linear_model import LinearRegression as dask_LinearRegression

from dask_ml.feature_extraction.text import CountVectorizer as dask_CountVectorizer
from sklearn.pipeline import Pipeline


pytest.skip(reason=f'pipes take too long', allow_module_level=True)

class TestDaskSetParams:


    @pytest.mark.parametrize('_refit',
        (False, 'accuracy', lambda x: 0), scope='class'
    )
    @pytest.mark.parametrize('_state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('junk_param',
        (0, 1, 3.14, None, True, 'trash', [0,1], (0, 1), min, lambda x: x)
    )
    def test_rejects_junk_params(
            self, junk_param, _state, _refit,
            dask_GSTCV_pipe_log_one_scorer_prefit,
            dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
            dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
            dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da,
            # _client
    ):

        if _state == 'prefit':
            _GSTCVDask_PIPE = dask_GSTCV_pipe_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da


        with pytest.raises(TypeError):
            _GSTCVDask_PIPE.set_params(junk_param)


    @pytest.mark.parametrize('_refit',
        (False, 'accuracy', lambda x: 0), scope='class'
    )
    @pytest.mark.parametrize('_state', ('prefit', 'postfit'))
    def test_rejects_invalid_params(
        self, _state, _refit,
        dask_GSTCV_pipe_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da,
        dask_est_xgb
    ):

        if _state == 'prefit':
            _GSTCVDask_PIPE = dask_GSTCV_pipe_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCVDask_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da


        # use this to reset the params to original state between tests and
        # at the end
        original_pipe_shallow_params = \
            deepcopy(_GSTCVDask_PIPE.get_params(deep=False))
        original_pipe_deep_params = \
            deepcopy(_GSTCVDask_PIPE.get_params(deep=True))


        # rejects_invalid_params ** * ** * ** * ** * ** * ** * ** * ** *
        # just check param names
        # invalid values for params should be caught at fit() by _validate()
        bad_params = dask_LinearRegression().get_params(deep=True)

        with pytest.raises(ValueError):
            _GSTCVDask_PIPE.set_params(**bad_params)
        # END rejects_invalid_params ** * ** * ** * ** * ** * ** * ** *


        # for shallow and deep pipe, just take all the params from
        # itself and verify accepts everything; change some of the params and
        # assert new settings are correct

        # test_accepts_good_params_shallow_pipe ** * ** * ** * ** * ** *

        good_params_pipe_shallow = _GSTCVDask_PIPE.get_params(deep=False)

        good_params_pipe_shallow['estimator'] = \
            Pipeline(steps=[('bag_of_words', dask_CountVectorizer()),
                            ('xgboost', dask_est_xgb)])
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
                          type(dask_est_xgb))
        assert _GSTCVDask_PIPE.param_grid == \
               {'bag_of_words__analyzer': ['word', 'char', 'char_wb'],
                'xgboost__max_depth': [3, 4, 5]}
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.n_jobs == 4
        assert _GSTCVDask_PIPE.cv == 5
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True

        # END shallow pipe ** *** ** *** ** *** ** *** ** *** ** *** **


        _GSTCVDask_PIPE.set_params(**original_pipe_shallow_params)
        _GSTCVDask_PIPE.set_params(**original_pipe_deep_params)


        # test_accepts_good_params_deep_pipe ** * ** * ** * ** * ** * ** * **

        good_params_pipe_deep = _GSTCVDask_PIPE.get_params(deep=True)

        good_params_pipe_deep['cv'] = 12
        good_params_pipe_deep['estimator__dask_StandardScaler__with_mean'] = True
        good_params_pipe_deep['estimator__dask_logistic__C'] = 1e-3
        good_params_pipe_deep['estimator__dask_logistic__max_iter'] = 10000
        good_params_pipe_deep['estimator__dask_logistic__n_jobs'] = None
        good_params_pipe_deep['n_jobs'] = 8
        good_params_pipe_deep['param_grid'] = {
            'dask_StandardScaler__with_std': [True, False],
            'dask_logistic__C': [0.0001, 0.001, 0.01]}
        good_params_pipe_deep['refit'] = False
        good_params_pipe_deep['return_train_score'] = True
        good_params_pipe_deep['scoring'] = 'balanced_accuracy'
        good_params_pipe_deep['verbose'] = 10

        _GSTCVDask_PIPE.set_params(**good_params_pipe_deep)

        assert _GSTCVDask_PIPE.cv == 12
        assert _GSTCVDask_PIPE.estimator.steps[0][0] == 'dask_StandardScaler'
        assert _GSTCVDask_PIPE.estimator.steps[0][1].with_mean is True
        assert _GSTCVDask_PIPE.estimator.steps[1][0] == 'dask_logistic'
        assert _GSTCVDask_PIPE.estimator.steps[1][1].C == 1e-3
        assert _GSTCVDask_PIPE.estimator.steps[1][1].max_iter == 10000
        assert _GSTCVDask_PIPE.estimator.steps[1][1].n_jobs == None
        assert _GSTCVDask_PIPE.n_jobs == 8
        assert _GSTCVDask_PIPE.param_grid == {
            'dask_StandardScaler__with_std': [True, False],
            'dask_logistic__C': [0.0001, 0.001, 0.01]
        }
        assert _GSTCVDask_PIPE.refit is False
        assert _GSTCVDask_PIPE.return_train_score is True
        assert _GSTCVDask_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCVDask_PIPE.verbose == 10

        # END accepts_good_params_deep_pipe ** * ** * ** * ** * ** * **


        _GSTCVDask_PIPE.set_params(**original_pipe_shallow_params)
        _GSTCVDask_PIPE.set_params(**original_pipe_deep_params)








