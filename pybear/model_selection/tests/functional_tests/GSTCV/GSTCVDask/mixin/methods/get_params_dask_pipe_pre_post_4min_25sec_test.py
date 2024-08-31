# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np


pytest.skip(reason=f'pipes take too long', allow_module_level=True)

# Tests GSTCV get_params against dask_GSCV get_params for shallow & deep


class TestDaskGetParams:


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, _refit, state, bad_deep,
        dask_GSTCV_pipe_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da
    ):

        if state == 'prefit':
            _GSTCV_PIPE = dask_GSTCV_pipe_log_one_scorer_prefit
        elif state == 'postfit':
            if _refit is False:
                _GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da

        with pytest.raises(ValueError):
            _GSTCV_PIPE.get_params(bad_deep)


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe(
        self, state, _refit,
        dask_GSCV_pipe_log_one_scorer_prefit,
        dask_GSCV_pipe_log_one_scorer_postfit_refit_false,
        dask_GSCV_pipe_log_one_scorer_postfit_refit_str,
        dask_GSCV_pipe_log_one_scorer_postfit_refit_fxn,
        dask_GSTCV_pipe_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da
    ):

        # test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _dask_GSCV_PIPE = dask_GSCV_pipe_log_one_scorer_prefit
            _dask_GSTCV_PIPE = dask_GSTCV_pipe_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _dask_GSCV_PIPE = dask_GSCV_pipe_log_one_scorer_postfit_refit_false
                _dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _dask_GSCV_PIPE = dask_GSCV_pipe_log_one_scorer_postfit_refit_str
                _dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _dask_GSCV_PIPE = dask_GSCV_pipe_log_one_scorer_postfit_refit_fxn
                _dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da


        assert _dask_GSCV_PIPE.estimator.steps[0][0] == 'dask_StandardScaler'
        assert _dask_GSCV_PIPE.estimator.steps[1][0] == 'dask_logistic'

        assert _dask_GSTCV_PIPE.estimator.steps[0][0] == 'dask_StandardScaler'
        assert _dask_GSTCV_PIPE.estimator.steps[1][0] == 'dask_logistic'

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(_dask_GSCV_PIPE.get_params(deep=False).keys())
        gstcv_shallow = list(_dask_GSTCV_PIPE.get_params(deep=False).keys())

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


        # test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        assert _dask_GSCV_PIPE.estimator.steps[0][0] == 'dask_StandardScaler'
        assert _dask_GSCV_PIPE.estimator.steps[1][0] == 'dask_logistic'

        assert _dask_GSTCV_PIPE.estimator.steps[0][0] == 'dask_StandardScaler'
        assert _dask_GSTCV_PIPE.estimator.steps[1][0] == 'dask_logistic'

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_deep = list(_dask_GSCV_PIPE.get_params(deep=True).keys())
        gstcv_deep = list(_dask_GSTCV_PIPE.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep
        assert 'verbose' in gstcv_deep
        # +2 for thresholds / verbose
        assert len(gstcv_deep) == len(daskgscv_deep) + 2

        assert len(daskgscv_deep) == 34
        assert len(gstcv_deep) == 36

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')

        assert np.array_equiv(daskgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *










