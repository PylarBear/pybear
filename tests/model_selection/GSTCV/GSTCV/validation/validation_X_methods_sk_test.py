# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from uuid import uuid4
import numpy as np
import pandas as pd
from copy import deepcopy




# this module tests all methods except score that accept a single X
# argument for handling X in good and a variety of bad conditions
# (non-numeric data, column count mismatch, feature name mismatch)



class TestSKGSTCVMethodsBesidesScore_XValidation:

    # methods & signatures (besides fit)
    # --------------------------
    # decision_function(X)
    # inverse_transform(Xt)
    # predict(X)
    # predict_log_proba(X)
    # predict_proba(X)
    # score_samples(X)
    # transform(X)



    @pytest.mark.parametrize('_fitted_format', ('np', 'df'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_X_format', ('np', 'df'))
    @pytest.mark.parametrize('_X_state', ('good', 'bad_features', 'bad_data'))
    def test_methods(self, _fitted_format, _scoring, _X_format, _X_state,
        _rows, _cols, COLUMNS,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd
    ):

        if _fitted_format == 'np':
            if _scoring == ['accuracy']:
                sk_GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                sk_GSTCV = sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np

        elif _fitted_format == 'df':
            if _scoring == ['accuracy']:
                sk_GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                sk_GSTCV = sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd

        if _X_state == 'good':
            X_sk = np.random.randint(0, 10, (_rows, _cols))
        elif _X_state == 'bad_features':
            X_sk = np.random.randint(0, 10, (_rows, 2*_cols))
        elif _X_state == 'bad_data':
            X_sk = np.random.choice(list('abcd'), (_rows, _cols), replace=True)


        if _X_format == 'df':

            columns = deepcopy(COLUMNS)

            if _X_state == 'bad_features':
                columns += [str(uuid4())[:4] for _ in range(_cols)]

            X_sk = pd.DataFrame(data=X_sk, columns=columns)


        # inverse_transform, score_samples, transform ** ** ** ** ** **

        # for all states of data, and np or pd
        for attr in ('inverse_transform', 'score_samples', 'transform'):

            with pytest.raises(AttributeError):
                getattr(sk_GSTCV, attr)(X_sk)

        # END inverse_transform, score_samples, transform test ** ** **


        # decision_function, predict_log_proba, predict_proba, predict ** ** **

        for attr in (
            'decision_function', 'predict_log_proba', 'predict_proba', 'predict'
        ):

            if _X_state == 'good':  # np or pd
                __ = getattr(sk_GSTCV, attr)(X_sk)
                assert isinstance(__, np.ndarray)
                if attr == 'predict':
                    assert __.dtype == np.uint8
                else:
                    assert __.dtype == np.float64
            elif _X_state == 'bad_features':
                if _X_format == 'np':
                    # this raises in the estimator, let is raise whatever
                    with pytest.raises(Exception):
                        getattr(sk_GSTCV, attr)(X_sk)
                elif _X_format == 'df':
                    # this raises in the estimator, let is raise whatever
                    with pytest.raises(Exception):
                        getattr(sk_GSTCV, attr)(X_sk)
            elif _X_state == 'bad_data':  # np or pd
                # this raises in the estimator, let is raise whatever
                with pytest.raises(Exception):
                    getattr(sk_GSTCV, attr)(X_sk)
        # END decision_function, predict_log_proba, predict_proba, predict ** **





