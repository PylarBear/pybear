# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from copy import deepcopy

from dask_ml.linear_model import LinearRegression as dask_LinearRegression





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
            dask_GSTCV_est_log_one_scorer_prefit,
            dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
            dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
            dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da,
            # _client
    ):

        if _state == 'prefit':
            _GSTCVDask = \
                dask_GSTCV_est_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da


        with pytest.raises(TypeError):
            _GSTCVDask.set_params(junk_param)



    @pytest.mark.parametrize('_refit',
        (False, 'accuracy', lambda x: 0), scope='class'
    )
    @pytest.mark.parametrize('_state', ('prefit', 'postfit'))
    def test_rejects_invalid_params(
        self, _state, _refit,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da,
        dask_est_xgb
    ):

        if _state == 'prefit':
            _GSTCVDask = dask_GSTCV_est_log_one_scorer_prefit
        elif _state == 'postfit':
            if _refit is False:
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                _GSTCVDask = \
                    dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da


        # use this to reset the params to original state between tests and
        # at the end
        original_single_est_params = deepcopy(_GSTCVDask.get_params(deep=True))


        # rejects_invalid_params ** * ** * ** * ** * ** * ** * ** * ** *
        # just check param names
        # invalid values for params should be caught at fit() by _validate()
        bad_params = dask_LinearRegression().get_params(deep=True)

        with pytest.raises(ValueError):
            _GSTCVDask.set_params(**bad_params)

        # END rejects_invalid_params ** * ** * ** * ** * ** * ** * ** *


        # for shallow and deep single est, just take all the params from
        # itself and verify accepts everything; change some of the params and
        # assert new settings are correct

        # accepts_good_params_shallow_single_est ** * ** * ** * ** * ** * ** *

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

        # END accepts_good_params_shallow_single_est ** * ** * ** * ** * **


        _GSTCVDask.set_params(**original_single_est_params)


        # test_accepts_good_params_deep_single_est ** * ** * ** * ** * ** *

        good_params_deep_single_est = _GSTCVDask.get_params(deep=True)

        good_params_deep_single_est['estimator__tol'] = 1e-6
        good_params_deep_single_est['estimator__C'] = 1e-3
        good_params_deep_single_est['estimator__fit_intercept'] = False
        good_params_deep_single_est['estimator__solver'] = 'saga'
        good_params_deep_single_est['estimator__max_iter'] = 10_000
        good_params_deep_single_est['estimator__n_jobs'] = 8

        _GSTCVDask.set_params(**good_params_deep_single_est)

        assert _GSTCVDask.estimator.tol == 1e-6
        assert _GSTCVDask.estimator.C == 1e-3
        assert _GSTCVDask.estimator.fit_intercept is False
        assert _GSTCVDask.estimator.solver == 'saga'
        assert _GSTCVDask.estimator.max_iter == 10_000
        assert _GSTCVDask.estimator.n_jobs == 8

        # END test_accepts_good_params_deep_single_est ** * ** * ** * ** *


        _GSTCVDask.set_params(**original_single_est_params)










