# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np




# Tests GSTCV get_params against dask_GSCV get_params for shallow & deep


class TestDaskGetParams:


    # 24_10_27 now inheriting get_params from BaseEstimator.
    # the below code previously tested the code that stood in stead of
    # BaseEstimator.get_params, keep this for backup.
    @pytest.mark.skip(reason=f"24_10_27, no bool val when inherit from BaseEstimator")
    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, _refit, state, bad_deep,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da
    ):

        if state == 'prefit':
            _GSTCV = dask_GSTCV_est_log_one_scorer_prefit
        elif state == 'postfit':
            if _refit is False:
                _GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da

        with pytest.raises(ValueError):
            _GSTCV.get_params(bad_deep)


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_single_estimator(
        self, state, _refit,
        dask_GSCV_est_log_one_scorer_prefit,
        dask_GSCV_est_log_one_scorer_postfit_refit_false,
        dask_GSCV_est_log_one_scorer_postfit_refit_str,
        dask_GSCV_est_log_one_scorer_postfit_refit_fxn,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da
    ):

        # test shallow single estimator ** * ** * ** * ** * ** * ** * **

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _dask_GSCV = dask_GSCV_est_log_one_scorer_prefit
            _dask_GSTCV = dask_GSTCV_est_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _dask_GSCV = dask_GSCV_est_log_one_scorer_postfit_refit_false
                _dask_GSTCV = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _dask_GSCV = dask_GSCV_est_log_one_scorer_postfit_refit_str
                _dask_GSTCV = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _dask_GSCV = dask_GSCV_est_log_one_scorer_postfit_refit_str
                _dask_GSTCV = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(_dask_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_dask_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +2 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')

        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # test shallow single estimator ** * ** * ** * ** * ** * ** * **

        # test deep single estimator ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_dask_GSCV.get_params(deep=True).keys())
        gstcv_deep = list(_dask_GSTCV.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        assert len(gstcv_deep) == len(skgscv_deep) + 2

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep single estimator ** * ** * ** * ** * ** * ** * **









