# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from uuid import uuid4
from copy import deepcopy
import numpy as np
import pandas as pd



# this module tests score for handling X & y in good and a variety of
# bad conditions (bad num / bad name features, non-num data, row mismatch)

# test array with column number mismatch with 'raises Exception' as
# opposed to specific errors, because this is the only condition that
# is not caught by GSTCV, but is raised inside the estimator passed to
# GSTCV (sklearn, whatever) and those exceptions might change. All other
# excepts are caught by GSTCV.



class TestSKScore_XyValidation:


    @pytest.mark.parametrize('_fit_format', ('array', 'df'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy']))
    @pytest.mark.parametrize('_X_format', ('array', 'df'))
    @pytest.mark.parametrize('_X_state',
        ('good', 'bad_features', 'bad_data', 'bad_rows')
    )
    @pytest.mark.parametrize('_y_state',
        ('good', 'bad_features', 'bad_data', 'bad_rows')
    )
    def test_scoring(self, _scoring, _fit_format, _X_format,
        _X_state, _y_state, _rows, _cols, COLUMNS, multilabel_y,
        non_binary_y, different_rows, non_num_X, partial_feature_names_exc,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd
    ):

        if _fit_format == 'array':
            if _scoring == ['accuracy']:
                sk_GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                sk_GSTCV = \
                    sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np
        elif _fit_format == 'df':
            if _scoring == ['accuracy']:
                sk_GSTCV = \
                    sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                sk_GSTCV = \
                    sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd


        # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _X_state == 'good':
            X_sk = np.random.randint(0, 10, (_rows, _cols))
        elif _X_state == 'bad_features':
            X_sk = np.random.randint(0, 10, (_rows, 2*_cols))
        elif _X_state == 'bad_data':
            X_sk = np.random.choice(list('abcd'), (_rows, _cols), replace=True)
        elif _X_state == 'bad_rows':
            X_sk = np.random.randint(0, 10, (2*_rows, _cols))


        if _X_format == 'df':

            columns = deepcopy(COLUMNS)

            if _X_state == 'bad_features':
                columns += [str(uuid4())[:4] for _ in range(_cols)]

            X_sk = pd.DataFrame(data=X_sk, columns=columns)

        # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _y_state == 'good':
            y_sk = np.random.randint(0, 2, (_rows, 1))
        elif _y_state == 'bad_features':
            y_sk = np.random.randint(0, 2, (_rows, 2))
        elif _y_state == 'bad_data':
            y_sk = np.random.choice(list('abcd'), (_rows, 1), replace=True)
        elif _y_state == 'bad_rows':
            y_sk = np.random.randint(0, 2, (2*_rows, 1))


        if _X_format == 'df':
            columns = ['y1']
            if _y_state == 'bad_features':
                columns += ['y2']

            y_sk = pd.DataFrame(data=y_sk, columns=columns)
        # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        if _X_state == 'bad_rows' and _y_state == 'bad_rows':
             pytest.skip(reason=f'skip when X & y have bad_rows (not bad then!)')


        attr = 'score'

        if _X_state == 'good':
            if _y_state == 'good':
                __ = getattr(sk_GSTCV, attr)(X_sk, y_sk)
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1
            elif _y_state == 'bad_features':
                with pytest.raises(ValueError, match=multilabel_y):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)
            elif _y_state == 'bad_data':
                with pytest.raises(ValueError, match=non_binary_y('GSTCV')):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)
            elif _y_state == 'bad_rows':
                exp_match = different_rows(y_sk.shape[0], X_sk.shape[0])
                with pytest.raises(ValueError, match=exp_match):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)

        elif _X_state == 'bad_features':
            if _y_state == 'good':
                if _fit_format == 'array':
                    with pytest.raises(Exception):
                        getattr(sk_GSTCV, attr)(X_sk, y_sk)
                elif _fit_format == 'dataframe':
                    with pytest.raises(ValueError) as exc:
                        getattr(sk_GSTCV, attr)(X_sk, y_sk)
                    assert partial_feature_names_exc in str(exc)
            elif _y_state == 'bad_features':
                with pytest.raises(ValueError, match=multilabel_y):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)
            elif _y_state == 'bad_data':
                with pytest.raises(ValueError, match=non_binary_y('GSTCV')):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)
            elif _y_state == 'bad_rows':
                exp_match = different_rows(y_sk.shape[0], X_sk.shape[0])
                with pytest.raises(ValueError, match=exp_match):
                    getattr(sk_GSTCV, attr)(X_sk, y_sk)

        elif _X_state == 'bad_data':
            # for all states of y
            with pytest.raises(ValueError, match=non_num_X):
                getattr(sk_GSTCV, attr)(X_sk, y_sk)

        elif _X_state == 'bad_rows':
            # for all states of y (except bad_rows, which is skipped)
            exp_match = different_rows(y_sk.shape[0], X_sk.shape[0])
            with pytest.raises(ValueError, match=exp_match):
                getattr(sk_GSTCV, attr)(X_sk, y_sk)






























