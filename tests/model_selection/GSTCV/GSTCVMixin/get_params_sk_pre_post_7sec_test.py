# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from GSTCV.conftest import param_grid_sk_log


class TestSKGetParams:


    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, bad_deep,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_prefit
    ):

        _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
        _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit

        with pytest.raises(ValueError):
            _GSTCV.get_params(bad_deep)

        with pytest.raises(ValueError):
            _GSTCV_PIPE.get_params(bad_deep)


    @pytest.mark.parametrize('_state,_refit',
        (('prefit', False), ('postfit', 'accuracy'), ('postfit', False)),
        scope='class'
    )
    def test_no_pipe(
        self, _state, _refit,
        sk_est_log,
        param_grid_sk_log,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
    ):

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *

        if _state == 'prefit':
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np

        # only test params' names, not values; GSTCV's defaults may be
        # different than skGSCV's

        gstcv_shallow = list(_GSTCV.get_params(deep=False).keys())

        exp_gstcv_shallow = {
            'cv': None,
            'error_score': 'raise',
            'estimator': sk_est_log,
            'n_jobs': None,
            'param_grid': param_grid_sk_log,
            'pre_dispatch': '2*n_jobs',
            'refit': True,
            'return_train_score': False,
            'scoring': 'accuracy',
            'thresholds': None,
            'verbose': 0
        }
        assert np.array_equiv(list(exp_gstcv_shallow), gstcv_shallow)
        for _param, _value in exp_gstcv_shallow.items():
            assert _param in gstcv_shallow
            # assert gstcv_shallow[_param] == _value   # pizza

        # pizza
        # exp_gstcv_dask_shallow = {
        #     'cache_cv': True,
        #     'cv': None,
        #     'estimator': sk_est_log(),
        #     'iid': True,
        #     'n_jobs': None,
        #     'error_score': 'raise',
        #     'param_grid': param_grid_sk_log,
        #     'refit': True,
        #     'return_train_score': False,
        #     'scheduler': None,
        #     'scoring': 'accuracy',
        #     'thresholds': None,
        #     'verbose': 0
        # }
        # assert np.array_equiv(list(exp_gstcv_dask_shallow), gstcv_dask_shallow)
        # for _param, _value in exp_gstcv_dask_shallow:
        #     assert _param in exp_gstcv_dask_shallow
            # assert exp_gstcv_dask_shallow[_param] == _value   # pizza

        # END test shallow no pipe ** * ** * ** * ** * ** * ** * ** * **

        # test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        gstcv_deep = list(_GSTCV.get_params(deep=True).keys())
        # gstcv_dask_deep = list(_GSTCVDask.get_params(deep=True).keys())

        # assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_state,_refit',
        (('prefit', False), ('postfit', 'accuracy'), ('postfit', False)),
        scope='class'
    )
    def test_pipe(
        self, _state, _refit,
        sk_GSCV_pipe_log_one_scorer_prefit,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_false,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_str,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
    ):

        # test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if _state == 'prefit':
            _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_false
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_str
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np


        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
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


        # test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV_PIPE.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV_PIPE.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep


        assert len(gstcv_deep) == len(skgscv_deep) + 1

        assert len(skgscv_deep) == 34
        assert len(gstcv_deep) == 35

        gstcv_deep.remove('thresholds')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *







