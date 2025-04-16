# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest



class TestDaskGSTCVMethodsPostFit:

    # methods & signatures (besides fit)
    # --------------------------
    # decision_function(X)
    # inverse_transform(Xt)
    # predict(X)
    # predict_log_proba(X)
    # predict_proba(X)
    # score(X, y=None, **params)
    # score_samples(X)
    # transform(X)
    # visualize(filename=None, format=None)



    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    def test_methods(
        self, _refit, X_da, X_ddf, y_da, generic_no_attribute_2, _no_refit,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da
    ):


        if _refit is False:
            GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da
        elif _refit == 'accuracy':
            GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
        elif callable(_refit):
            GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da


        X_dask = X_da


        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "visualize is not implemented in GSTCVDask"
        with pytest.raises(NotImplementedError, match=exc_info):
            getattr(GSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















