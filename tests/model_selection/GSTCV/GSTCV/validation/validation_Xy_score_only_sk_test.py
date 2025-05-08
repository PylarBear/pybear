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



class TestSKScore_XyValidation:


    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(
        self, junk_X, y_np, sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
    ):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np.fit(junk_X, y_np)


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(
        self, X_np, junk_y, sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
    ):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np.fit(X_np, junk_y)



    @pytest.mark.parametrize('_fitted_format', ('np', 'df'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy']))
    @pytest.mark.parametrize('_container', ('np', 'df'))
    @pytest.mark.parametrize('_X_state',
        ('good', 'bad_features', 'bad_data', 'bad_rows')
    )
    @pytest.mark.parametrize('_y_state',
        ('good', 'bad_features', 'bad_data', 'bad_rows')
    )
    def test_scoring(self, _scoring, _fitted_format, _container,
        _X_state, _y_state, _rows, _cols, COLUMNS,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_pd,
        sk_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_pd
    ):

        if _X_state == 'good' and _y_state == 'good':
            pytest.skip(reason=f'other tests already show fit on good works')

        if _X_state == 'bad_rows' and _y_state == 'bad_rows':
             pytest.skip(reason=f'skip when X & y have bad_rows (not bad then!)')

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


        # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _X_state == 'good':
            X_sk = np.random.randint(0, 10, (_rows, _cols))
        elif _X_state == 'bad_features':
            X_sk = np.random.randint(0, 10, (_rows, 2*_cols))
        elif _X_state == 'bad_data':
            X_sk = np.random.choice(list('abcd'), (_rows, _cols), replace=True)
        elif _X_state == 'bad_rows':
            X_sk = np.random.randint(0, 10, (2*_rows, _cols))


        if _container == 'df':

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


        if _container == 'df':
            columns = ['y1']
            if _y_state == 'bad_features':
                columns += ['y2']

            y_sk = pd.DataFrame(data=y_sk, columns=columns)
        # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        if _X_state == 'good':
            if _y_state == 'good':
                __ = getattr(sk_GSTCV, 'score')(X_sk, y_sk)
                assert isinstance(__, float)
                assert 0 <= __ <= 1
            elif _y_state == 'bad_features':
                # this is raised by GSTCV in _val_X_y
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)
            elif _y_state == 'bad_data':
                # this is raised by _val_X_y for not in [0,1]
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)
            elif _y_state == 'bad_rows':
                # this is raised by GSTCV in _val_X_y
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)

        elif _X_state == 'bad_features':
            if _y_state == 'good':
                if _fitted_format == 'np':
                    # this is raised by the estimator, let it raise whatever
                    with pytest.raises(Exception):
                        getattr(sk_GSTCV, 'score')(X_sk, y_sk)
                elif _fitted_format == 'df':
                    # this is raised by the estimator, let it raise whatever
                    with pytest.raises(Exception):
                        getattr(sk_GSTCV, 'score')(X_sk, y_sk)
            elif _y_state == 'bad_features':
                # this is raised by GSTCV in _val_X_y
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)
            elif _y_state == 'bad_data':
                # this is raised by GSTCV in _val_X_y
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)
            elif _y_state == 'bad_rows':
                # this is raised by GSTCV in _val_X_y
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'score')(X_sk, y_sk)

        elif _X_state == 'bad_data':
            # for all states of y
            # this is raised by the estimator, let it raise whatever
            with pytest.raises(Exception):
                getattr(sk_GSTCV, 'score')(X_sk, y_sk)

        elif _X_state == 'bad_rows':
            # for all states of y (except bad_rows, which is skipped)
            # this is raised by GSTCV in _val_X_y
            with pytest.raises(ValueError):
                getattr(sk_GSTCV, 'score')(X_sk, y_sk)





