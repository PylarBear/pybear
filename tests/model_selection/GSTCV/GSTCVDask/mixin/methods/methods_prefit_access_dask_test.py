# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest



class TestDaskGSTCVMethodsPreFit:

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
        self, _refit, X_da, y_da, COLUMNS, _no_refit,
        generic_no_attribute_2, _not_fitted,
        dask_GSTCV_est_log_one_scorer_prefit
    ):


        DaskGSTCV = dask_GSTCV_est_log_one_scorer_prefit


        DaskGSTCV.set_params(refit=_refit)


        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _not_fitted('GSTCVDask')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **





