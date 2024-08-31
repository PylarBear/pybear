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


    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    def test_methods(
        self, _refit, _scoring, X_da, y_da, COLUMNS, _no_refit,
        generic_no_attribute_2, _not_fitted,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_prefit,
        dask_GSTCV_est_log_two_scorers_prefit,
        dask_GSTCV_pipe_log_two_scorers_prefit
    ):

        if _scoring == ['accuracy']:
            DaskGSTCV = dask_GSTCV_est_log_one_scorer_prefit
            DaskGSTCV_PIPE = dask_GSTCV_pipe_log_one_scorer_prefit

        elif _scoring == ['accuracy', 'balanced_accuracy']:
            DaskGSTCV = dask_GSTCV_est_log_two_scorers_prefit
            DaskGSTCV_PIPE = dask_GSTCV_pipe_log_two_scorers_prefit


        DaskGSTCV.set_params(refit=_refit)
        DaskGSTCV_PIPE.set_params(refit=_refit)


        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'decision_function')(X_da)

            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV_PIPE, 'decision_function')(X_da)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'decision_function')(X_da)

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV_PIPE, 'decision_function')(X_da)

        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(DaskGSTCV, 'get_metadata_routing')()

        with pytest.raises(NotImplementedError):
            getattr(DaskGSTCV_PIPE, 'get_metadata_routing')()

        # END get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'inverse_transform')(X_da)

        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV_PIPE, 'inverse_transform')(X_da)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'predict')(X_da)

            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV_PIPE, 'predict')(X_da)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'predict')(X_da)

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV_PIPE, 'predict')(X_da)

        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'predict_log_proba')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'predict_log_proba')(X_da)

        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV_PIPE, 'predict_log_proba')(X_da)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'predict_proba')(X_da)

            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV_PIPE, 'predict_proba')(X_da)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'predict_proba')(X_da)

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV_PIPE, 'predict_proba')(X_da)

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV, 'score')(X_da, y_da)

            with pytest.raises(AttributeError, match=exc_info):
                getattr(DaskGSTCV_PIPE, 'score')(X_da, y_da)

        elif _refit == 'accuracy' or callable(_refit):

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV, 'score')(X_da, y_da)

            with pytest.raises(AttributeError, match=_not_fitted('GSTCVDask')):
                getattr(DaskGSTCV_PIPE, 'score')(X_da, y_da)

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'score_samples')(X_da)

        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV_PIPE, 'score_samples')(X_da)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'transform')(X_da)

        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV_PIPE, 'transform')(X_da)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = _not_fitted('GSTCVDask')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV, 'visualize')(filename=None, format=None)

        with pytest.raises(AttributeError, match=exc_info):
            getattr(DaskGSTCV_PIPE, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **





