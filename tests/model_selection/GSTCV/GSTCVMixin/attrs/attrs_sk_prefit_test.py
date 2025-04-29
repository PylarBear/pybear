# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest



class TestSKAttrsPreFit:

    # pre-fit, all attrs should not be available and should except

    @pytest.mark.parametrize('attr',
        ('cv_results_', 'best_estimator_', 'best_index_', 'scorer_', 'n_splits_',
         'refit_time_', 'multimetric_', 'feature_names_in_', 'best_threshold_',
         'best_score_', 'best_params_'
         )
    )
    def test_attrs_1(self, sk_GSTCV_est_log_one_scorer_prefit,
        generic_no_attribute_1, attr
    ):

        sk_GSTCV_prefit = sk_GSTCV_est_log_one_scorer_prefit

        with pytest.raises(
            AttributeError,
            match=generic_no_attribute_1('GSTCV', attr)
        ):
            getattr(sk_GSTCV_prefit, attr)


    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    def test_classes__n_features_in(self, _scoring, _refit_false, _not_fitted,
        generic_no_attribute_1, sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_two_scorers_prefit,
    ):

        if _scoring == ['accuracy']:
            sk_GSTCV_prefit = sk_GSTCV_est_log_one_scorer_prefit

        elif _scoring == ['accuracy', 'balanced_accuracy']:
            sk_GSTCV_prefit = sk_GSTCV_est_log_two_scorers_prefit


        with pytest.raises(AttributeError, match=_refit_false('GSTCV')):
            sk_GSTCV_prefit.classes_


        with pytest.raises(AttributeError):
            sk_GSTCV_prefit.n_features_in_







