# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy

from sklearn.linear_model import LinearRegression as sk_LinearRegression

from sklearn.feature_extraction.text import CountVectorizer as sk_CountVectorizer
from sklearn.pipeline import Pipeline



class TestSKSetParams:


    @pytest.mark.parametrize('_refit',
        (False, 'accuracy', lambda x: 0), scope='class'
    )
    @pytest.mark.parametrize('_state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('junk_param',
        (0, 1, 3.14, None, True, 'trash', [0,1], (0, 1), min, lambda x: x)
    )
    def test_rejects_junk_params(
        self, junk_param, _state, _refit,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        if _state == 'prefit':
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


        with pytest.raises(TypeError):
            _GSTCV.set_params(junk_param)

        with pytest.raises(TypeError):
            _GSTCV_PIPE.set_params(junk_param)



    @pytest.mark.parametrize('_refit',
        (False, 'accuracy', lambda x: 0), scope='class'
    )
    @pytest.mark.parametrize('_state', ('prefit', 'postfit'))
    def test_accuracy(
        self, _state, _refit,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_est_log
    ):

        if _state == 'prefit':
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
                _GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


        # use this to reset the params to original state between tests and
        # at the end
        original_no_pipe_params = deepcopy(_GSTCV.get_params(deep=True))
        original_pipe_shallow_params = deepcopy(_GSTCV_PIPE.get_params(deep=False))
        original_pipe_deep_params = deepcopy(_GSTCV_PIPE.get_params(deep=True))


        # rejects_invalid_params ** * ** * ** * ** * ** * ** * ** * ** *
        # just check param names
        # invalid values for params should be caught at fit() by _validate()  # pizza fix this!
        bad_params = sk_LinearRegression().get_params(deep=True)

        with pytest.raises(ValueError):
            _GSTCV.set_params(**bad_params)

        with pytest.raises(ValueError):
            _GSTCV_PIPE.set_params(**bad_params)
        # END rejects_invalid_params ** * ** * ** * ** * ** * ** * ** *


        # for shallow and deep no pipe / pipe, just take all the params from
        # itself and verify accepts everything; change some of the params and
        # assert new settings are correct

        # accepts_good_params_shallow_no_pipe ** * ** * ** * ** * ** * **

        good_params_shallow = _GSTCV.get_params(deep=False)

        good_params_shallow['thresholds'] = [0.1, 0.5, 0.9]
        good_params_shallow['scoring'] = 'balanced_accuracy'
        good_params_shallow['n_jobs'] = 4
        good_params_shallow['cv'] = 8
        good_params_shallow['refit'] = False
        good_params_shallow['verbose'] = 10
        good_params_shallow['return_train_score'] = True

        _GSTCV.set_params(**good_params_shallow)

        assert _GSTCV.thresholds == [0.1, 0.5, 0.9]
        assert _GSTCV.scoring == 'balanced_accuracy'
        assert _GSTCV.n_jobs == 4
        assert _GSTCV.cv == 8
        assert _GSTCV.refit is False
        assert _GSTCV.verbose == 10
        assert _GSTCV.return_train_score is True

        # END accepts_good_params_shallow_no_pipe ** * ** * ** * ** * **


        _GSTCV.set_params(**original_no_pipe_params)


        # accepts_good_params_deep_no_pipe ** * ** * ** * ** * ** * ** *

        good_params_deep_no_pipe = _GSTCV.get_params(deep=True)

        good_params_deep_no_pipe['estimator__tol'] = 1e-6
        good_params_deep_no_pipe['estimator__C'] = 1e-3
        good_params_deep_no_pipe['estimator__fit_intercept'] = False
        good_params_deep_no_pipe['estimator__solver'] = 'saga'
        good_params_deep_no_pipe['estimator__max_iter'] = 10_000
        good_params_deep_no_pipe['estimator__n_jobs'] = 8

        _GSTCV.set_params(**good_params_deep_no_pipe)

        assert _GSTCV.estimator.tol == 1e-6
        assert _GSTCV.estimator.C == 1e-3
        assert _GSTCV.estimator.fit_intercept is False
        assert _GSTCV.estimator.solver == 'saga'
        assert _GSTCV.estimator.max_iter == 10_000
        assert _GSTCV.estimator.n_jobs == 8

        # END accepts_good_params_deep_no_pipe ** * ** * ** * ** * ** *


        _GSTCV.set_params(**original_no_pipe_params)


        # accepts_good_params_shallow_pipe ** * ** * ** * ** * ** * ** *

        good_params_pipe_shallow = _GSTCV_PIPE.get_params(deep=False)

        good_params_pipe_shallow['estimator'] = \
            Pipeline(steps=[('bag_of_words', sk_CountVectorizer()),
                            ('sk_logistic', sk_est_log)])
        good_params_pipe_shallow['param_grid'] = \
            {'C': [1e-6, 1e-5, 1e-4], 'solver': ['saga', 'lbfgs']}
        good_params_pipe_shallow['scoring'] = 'balanced_accuracy'
        good_params_pipe_shallow['n_jobs'] = 4
        good_params_pipe_shallow['cv'] = 5
        good_params_pipe_shallow['refit'] = False
        good_params_pipe_shallow['return_train_score'] = True

        _GSTCV_PIPE.set_params(**good_params_pipe_shallow)

        assert isinstance(_GSTCV_PIPE.estimator, Pipeline)
        assert isinstance(_GSTCV_PIPE.estimator.steps[0][1], sk_CountVectorizer)
        assert isinstance(_GSTCV_PIPE.estimator.steps[1][1], type(sk_est_log))
        assert _GSTCV_PIPE.param_grid == \
               {'C': [1e-6, 1e-5, 1e-4], 'solver': ['saga', 'lbfgs']}
        assert _GSTCV_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCV_PIPE.n_jobs == 4
        assert _GSTCV_PIPE.cv == 5
        assert _GSTCV_PIPE.refit is False
        assert _GSTCV_PIPE.return_train_score is True

        # END accepts_good_params_shallow_pipe ** * ** * ** * ** * ** *


        _GSTCV_PIPE.set_params(**original_pipe_shallow_params)
        _GSTCV_PIPE.set_params(**original_pipe_deep_params)


        # accepts_good_params_deep_pipe ** * ** * ** * ** * ** * ** * **

        good_params_pipe_deep = _GSTCV_PIPE.get_params(deep=True)

        good_params_pipe_deep['cv'] = 12
        good_params_pipe_deep['estimator__sk_StandardScaler__with_mean'] = True
        good_params_pipe_deep['estimator__sk_logistic__C'] = 1e-3
        good_params_pipe_deep['estimator__sk_logistic__max_iter'] = 10000
        good_params_pipe_deep['estimator__sk_logistic__n_jobs'] = None
        good_params_pipe_deep['n_jobs'] = 8
        good_params_pipe_deep['param_grid'] = {
            'sk_StandardScaler__with_std': [True, False],
            'sk_logistic__C': [0.0001, 0.001, 0.01]}
        good_params_pipe_deep['refit'] = False
        good_params_pipe_deep['return_train_score'] = True
        good_params_pipe_deep['scoring'] = 'balanced_accuracy'
        good_params_pipe_deep['verbose'] = 10

        _GSTCV_PIPE.set_params(**good_params_pipe_deep)

        assert _GSTCV_PIPE.cv == 12
        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSTCV_PIPE.estimator.steps[0][1].with_mean is True
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'
        assert _GSTCV_PIPE.estimator.steps[1][1].C == 1e-3
        assert _GSTCV_PIPE.estimator.steps[1][1].max_iter == 10000
        assert _GSTCV_PIPE.estimator.steps[1][1].n_jobs == None
        assert _GSTCV_PIPE.n_jobs == 8
        assert _GSTCV_PIPE.param_grid == {
            'sk_StandardScaler__with_std': [True, False],
            'sk_logistic__C': [0.0001, 0.001, 0.01]
        }
        assert _GSTCV_PIPE.refit is False
        assert _GSTCV_PIPE.return_train_score is True
        assert _GSTCV_PIPE.scoring == 'balanced_accuracy'
        assert _GSTCV_PIPE.verbose == 10

        # END accepts_good_params_deep_pipe ** * ** * ** * ** * ** * **


        _GSTCV_PIPE.set_params(**original_pipe_shallow_params)
        _GSTCV_PIPE.set_params(**original_pipe_deep_params)








