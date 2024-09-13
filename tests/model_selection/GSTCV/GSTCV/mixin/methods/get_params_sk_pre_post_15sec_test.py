# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np




# Tests GSTCV get_params against sk_GSCV get_params for shallow & deep


class TestSKGetParams:


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, _refit, state, bad_deep,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        if state == 'prefit':
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit
        elif state == 'postfit':
            if _refit is False:
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


        with pytest.raises(ValueError):
            _GSTCV.get_params(bad_deep)

        with pytest.raises(ValueError):
            _GSTCV_PIPE.get_params(bad_deep)


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_no_pipe(
        self, state, _refit,
        sk_GSCV_est_log_one_scorer_prefit,
        sk_GSCV_est_log_one_scorer_postfit_refit_false,
        sk_GSCV_est_log_one_scorer_postfit_refit_str,
        sk_GSCV_est_log_one_scorer_postfit_refit_fxn,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _GSCV = sk_GSCV_est_log_one_scorer_prefit
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_false
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_str
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_str
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np

        # only test params' names, not values; GSTCV's defaults may be
        # different than skGSCV's

        skgscv_shallow = list(_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        assert len(gstcv_shallow) == len(skgscv_shallow)

        gstcv_shallow.remove('thresholds')
        skgscv_shallow.remove('pre_dispatch')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow no pipe ** * ** * ** * ** * ** * ** * ** * **

        # test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        assert len(gstcv_deep) == len(skgscv_deep)

        gstcv_deep.remove('thresholds')
        skgscv_deep.remove('pre_dispatch')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe(
        self, state, _refit,
        sk_GSCV_pipe_log_one_scorer_prefit,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_false,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_str,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_fxn,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        # test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_false
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_str
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_fxn
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


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
        assert len(gstcv_shallow) == len(skgscv_shallow)

        assert len(skgscv_shallow) == 10
        assert len(gstcv_shallow) == 10

        gstcv_shallow.remove('thresholds')
        skgscv_shallow.remove('pre_dispatch')

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


        assert len(gstcv_deep) == len(skgscv_deep)

        assert len(skgscv_deep) == 33
        assert len(gstcv_deep) == 33

        gstcv_deep.remove('thresholds')
        skgscv_deep.remove('pre_dispatch')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

























