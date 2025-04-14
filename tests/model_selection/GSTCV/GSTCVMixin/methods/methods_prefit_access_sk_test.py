# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest



class TestSKGSTCVMethodsPreFit:

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



    # '_refit' mark must have False last! set_params(refit=xxx) is mutating
    # session fixtures. we need to leave the session fixtures at the starting
    # value for 'refit' in the 'sk_gscv_init_params', which is False.
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_refit', ('accuracy', lambda x: 0, False))
    def test_methods(
        self, _refit, _scoring, X_np, y_np, COLUMNS, _no_refit,
        generic_no_attribute_2, _not_fitted,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_two_scorers_prefit,
    ):

        if _scoring == ['accuracy']:
            GSTCV = sk_GSTCV_est_log_one_scorer_prefit

        elif _scoring == ['accuracy', 'balanced_accuracy']:
            GSTCV = sk_GSTCV_est_log_two_scorers_prefit


        GSTCV.set_params(refit=_refit)


        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'decision_function')(X_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'decision_function')(X_np)

        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(GSTCV, 'get_metadata_routing')()

        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'inverse_transform')(X_np)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict')(X_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict')(X_np)

        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_log_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict_log_proba')(X_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict_log_proba')(X_np)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'predict_proba')(X_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'predict_proba')(X_np)

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(GSTCV, 'score')(X_np, y_np)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCV')):
                getattr(GSTCV, 'score')(X_np, y_np)

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'score_samples')(X_np)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'transform')(X_np)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "'GSTCV' object has no attribute 'visualize'"
        with pytest.raises(AttributeError, match=exc_info):
            getattr(GSTCV, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **





