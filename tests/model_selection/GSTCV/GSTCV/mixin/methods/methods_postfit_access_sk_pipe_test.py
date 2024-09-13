# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np



class TestSKGSTCVMethodsPostFit:

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


    @pytest.mark.parametrize('_format', ('array', 'df'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy']))
    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    def test_methods(
        self, _refit, _scoring, _format, X_np, X_pd,
        y_np, generic_no_attribute_2, _no_refit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_np
    ):

        if _scoring == ['accuracy']:
            if _refit is False:
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
        elif _scoring == ['accuracy', 'balanced_accuracy']:
            if _refit is False:
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_np
            elif callable(_refit):
                sk_GSTCV_PIPE = \
                    sk_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_np

        X_sk = X_np if _format == 'array' else X_pd

        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:
            exc_info = _no_refit('GSTCV', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(sk_GSTCV_PIPE, 'decision_function')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):
            __ = getattr(sk_GSTCV_PIPE, 'decision_function')(X_sk)
            assert isinstance(__, np.ndarray)
            assert __.dtype == np.float64
        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(sk_GSTCV_PIPE, 'get_metadata_routing')()

        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(sk_GSTCV_PIPE, 'inverse_transform')(X_sk)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(sk_GSTCV_PIPE, 'predict')(X_sk)

        elif _refit == 'accuracy':

            __ = getattr(sk_GSTCV_PIPE, 'predict')(X_sk)
            assert isinstance(__, np.ndarray)
            assert __.dtype == np.uint8

        elif callable(_refit):

            # this is to accommodate lack of threshold when > 1 scorer
            if isinstance(_scoring, list) and len(_scoring) > 1:
                with pytest.raises(AttributeError):
                    getattr(sk_GSTCV_PIPE, 'predict')(X_sk)
            else:
                __ = getattr(sk_GSTCV_PIPE, 'predict')(X_sk)
                assert isinstance(__, np.ndarray)
                assert __.dtype == np.uint8


        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_log_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(sk_GSTCV_PIPE, 'predict_log_proba')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            __ = getattr(sk_GSTCV_PIPE, 'predict_log_proba')(X_sk)
            assert isinstance(__, np.ndarray)
            assert __.dtype == np.float64

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(sk_GSTCV_PIPE, 'predict_proba')(X_sk)

        elif _refit == 'accuracy' or callable(_refit):

            __ = getattr(sk_GSTCV_PIPE, 'predict_proba')(X_sk)
            assert isinstance(__, np.ndarray)
            assert __.dtype == np.float64

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCV', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(sk_GSTCV_PIPE, 'score')(X_sk, y_np)

        elif _refit == 'accuracy':

            __ = getattr(sk_GSTCV_PIPE, 'score')(X_sk, y_np)
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1

        elif callable(_refit):
            __ = getattr(sk_GSTCV_PIPE, 'score')(X_sk, y_np)
            if not isinstance(_scoring, list) or len(_scoring) == 1:
                # if refit fxn & one scorer, score is always returned
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1
            else:
                # if refit fxn & >1 scorer, refit fxn is returned
                assert callable(__)
                cvr = sk_GSTCV_PIPE.cv_results_
                assert isinstance(__(cvr), int) # refit(cvr) returns best_index_
                # best_index_ must be in range of the rows in cvr
                assert __(cvr) >= 0
                assert __(cvr) < len(cvr[list(cvr)[0]])

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(sk_GSTCV_PIPE, 'score_samples')(X_sk)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCV', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(sk_GSTCV_PIPE, 'transform')(X_sk)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "'GSTCV' object has no attribute 'visualize'"
        with pytest.raises(AttributeError, match=exc_info):
            getattr(sk_GSTCV_PIPE, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















