# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


class TestDaskAttrsPreFit:

    # pre-fit, all attrs should not be available and should except.

    @pytest.mark.parametrize('attr',
        ('cv_results_', 'best_estimator_', 'best_index_', 'scorer_', 'n_splits_',
         'refit_time_', 'multimetric_', 'feature_names_in_', 'best_threshold_',
         'best_score_', 'best_params_'
         )
    )
    def test_attrs_1(self, dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_prefit, generic_no_attribute_1, attr
    ):

        dask_GSTCV_prefit = dask_GSTCV_est_log_one_scorer_prefit
        dask_GSTCV_PIPE_prefit = dask_GSTCV_pipe_log_one_scorer_prefit

        with pytest.raises(
            AttributeError,
            match=generic_no_attribute_1('GSTCVDask', attr)
        ):
            getattr(dask_GSTCV_prefit, attr)

        with pytest.raises(
            AttributeError,
            match=generic_no_attribute_1('GSTCVDask', attr)
        ):
            getattr(dask_GSTCV_PIPE_prefit, attr)


    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    def test_classes__n_features_in(self, _scoring, _refit_false, _not_fitted,
        generic_no_attribute_1, generic_no_attribute_3,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_pipe_log_one_scorer_prefit,
        dask_GSTCV_est_log_two_scorers_prefit,
        dask_GSTCV_pipe_log_two_scorers_prefit
    ):

        if _scoring == ['accuracy']:
            dask_GSTCV_prefit = dask_GSTCV_est_log_one_scorer_prefit
            dask_GSTCV_PIPE_prefit = dask_GSTCV_pipe_log_one_scorer_prefit

        elif _scoring == ['accuracy', 'balanced_accuracy']:
            dask_GSTCV_prefit = dask_GSTCV_est_log_two_scorers_prefit
            dask_GSTCV_PIPE_prefit = dask_GSTCV_pipe_log_two_scorers_prefit


        with pytest.raises(AttributeError, match=_refit_false('GSTCVDask')):
            dask_GSTCV_prefit.classes_

        with pytest.raises(AttributeError, match=_refit_false('GSTCVDask')):
            dask_GSTCV_PIPE_prefit.classes_


        exp_match = generic_no_attribute_3("GSTCVDask", 'n_features_in_')
        with pytest.raises(AttributeError, match=exp_match):
            dask_GSTCV_prefit.n_features_in_

        with pytest.raises(AttributeError, match=exp_match):
            dask_GSTCV_PIPE_prefit.n_features_in_












