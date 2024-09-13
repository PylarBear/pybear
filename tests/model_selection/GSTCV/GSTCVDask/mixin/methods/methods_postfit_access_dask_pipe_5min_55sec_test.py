# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import dask.array as da


pytest.skip(reason=f'pipes take too long', allow_module_level=True)

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


    @pytest.mark.parametrize('_format', ('array', 'df'))
    @pytest.mark.parametrize('_scoring',
         (['accuracy'], ['accuracy', 'balanced_accuracy']))
    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    def test_methods(
        self, _refit, _scoring, _format, X_da, X_ddf,
        y_da, generic_no_attribute_2, _no_refit,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_da
    ):

        if _scoring == ['accuracy']:
            if _refit is False:
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
            elif callable(_refit):
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da
        elif _scoring == ['accuracy', 'balanced_accuracy']:
            if _refit is False:
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_da
            elif _refit == 'accuracy':
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_da
            elif callable(_refit):
                dask_GSTCV_PIPE = \
                    dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_da

        X_dask = X_da if _format == 'array' else X_ddf

        # decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:
            exc_info = _no_refit('GSTCVDask', True, 'decision_function')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(dask_GSTCV_PIPE, 'decision_function')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):
            __ = getattr(dask_GSTCV_PIPE, 'decision_function')(X_dask)
            assert isinstance(__, da.core.Array)
            assert __.dtype == np.float64
        # END decision_function ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **

        with pytest.raises(NotImplementedError):
            getattr(dask_GSTCV_PIPE, 'get_metadata_routing')()

        # get_metadata_routing ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'inverse_transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(dask_GSTCV_PIPE, 'inverse_transform')(X_dask)

        # END inverse_transform_test ** ** ** ** ** ** ** ** ** ** ** **

        # predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(dask_GSTCV_PIPE, 'predict')(X_dask)

        elif _refit == 'accuracy':

            __ = getattr(dask_GSTCV_PIPE, 'predict')(X_dask)
            assert isinstance(__, da.core.Array)
            assert __.dtype == np.uint8

        elif callable(_refit):

            # this is to accommodate lack of threshold when > 1 scorer
            if isinstance(_scoring, list) and len(_scoring) > 1:
                with pytest.raises(AttributeError):
                    getattr(dask_GSTCV_PIPE, 'predict')(X_dask)
            else:
                __ = getattr(dask_GSTCV_PIPE, 'predict')(X_dask)
                assert isinstance(__, da.core.Array)
                assert __.dtype == np.uint8


        # END predict ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'predict_log_proba')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(dask_GSTCV_PIPE, 'predict_log_proba')(X_dask)

        # END predict_log_proba ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'predict_proba')
            with pytest.raises(AttributeError, match=exc_info):
                getattr(dask_GSTCV_PIPE, 'predict_proba')(X_dask)

        elif _refit == 'accuracy' or callable(_refit):

            __ = getattr(dask_GSTCV_PIPE, 'predict_proba')(X_dask)
            assert isinstance(__, da.core.Array)
            assert __.dtype == np.float64

        # END predict_proba ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is False:

            exc_info = _no_refit('GSTCVDask', True, 'score')
            with pytest.raises(AttributeError, match=exc_info):
                getattr( dask_GSTCV_PIPE, 'score')(X_dask, y_da)

        elif _refit == 'accuracy':

            __ = getattr( dask_GSTCV_PIPE, 'score')(X_dask, y_da)
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1

        elif callable(_refit):
            __ = getattr(dask_GSTCV_PIPE, 'score')(X_dask, y_da)
            if not isinstance(_scoring, list) or len(_scoring) == 1:
                # if refit fxn & one scorer, score is always returned
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1
            else:
                # if refit fxn & >1 scorer, refit fxn is returned
                assert callable(__)
                cvr = dask_GSTCV_PIPE.cv_results_
                assert isinstance(__(cvr), int) # refit(cvr) returns best_index_
                # best_index_ must be in range of the rows in cvr
                assert __(cvr) >= 0
                assert __(cvr) < len(cvr[list(cvr)[0]])

        # END score ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'score_samples')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(dask_GSTCV_PIPE, 'score_samples')(X_dask)

        # END score_samples ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = generic_no_attribute_2('GSTCVDask', 'transform')
        with pytest.raises(AttributeError, match=exc_info):
            getattr(dask_GSTCV_PIPE, 'transform')(X_dask)

        # END transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        exc_info = "visualize is not implemented in GSTCVDask"
        with pytest.raises(NotImplementedError, match=exc_info):
            getattr(dask_GSTCV_PIPE, 'visualize')(filename=None, format=None)

        # END visualize ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



















