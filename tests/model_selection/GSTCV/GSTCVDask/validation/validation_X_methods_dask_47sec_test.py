# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from uuid import uuid4
import numpy as np
import dask.array as da
import dask.dataframe as ddf
from copy import deepcopy



# this module tests all methods except score that accept a single X
# argument for handling X in good and a variety of bad conditions
# (non-numeric data, column count mismatch, feature name mismatch)



class TestDaskGSTCVMethodsBesidesScore_XValidation:

    # methods & signatures (besides fit)
    # --------------------------
    # decision_function(X)
    # inverse_transform(Xt)
    # predict(X)
    # predict_log_proba(X)
    # predict_proba(X)
    # score_samples(X)
    # transform(X)


    # dask KFold cant take df
    @pytest.mark.parametrize('_fitted_format', ('da', )) # 'df'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_X_format', ('da', )) # 'df'))
    @pytest.mark.parametrize('_X_state', ('good', 'bad_features', 'bad_data'))
    def test_methods(self, _fitted_format, _scoring, _X_format, _X_state,
        _rows, _cols, COLUMNS, _client,   # need client or client closed error
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da,
        # dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf,
        # dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf,
    ):

        if _fitted_format == 'da':
            if _scoring == ['accuracy']:
                dask_GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                dask_GSTCV = dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da

        elif _fitted_format == 'df':
            if _scoring == ['accuracy']:
                dask_GSTCV = dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf
            elif _scoring == ['accuracy', 'balanced_accuracy']:
                dask_GSTCV = dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf

        if _X_state == 'good':
            X_dask = da.random.randint(0, 10, (_rows, _cols))
        elif _X_state == 'bad_features':
            X_dask = da.random.randint(0, 10, (_rows, 2*_cols))
        elif _X_state == 'bad_data':
            X_dask = da.random.choice(list('abcd'), (_rows, _cols), replace=True)


        if _X_format == 'df':

            columns = deepcopy(COLUMNS)

            if _X_state == 'bad_features':
                columns += [str(uuid4())[:4] for _ in range(_cols)]

            X_dask = ddf.from_dask_array(X_dask, columns=columns)


        # inverse_transform, score_samples, transform ** ** ** ** ** **

        # for all states of data, and da or df
        for attr in ('inverse_transform', 'score_samples', 'transform'):

            with pytest.raises(AttributeError):
                getattr(dask_GSTCV, attr)(X_dask)

        # END inverse_transform, score_samples, transform test ** ** **


        # decision_function, predict_log_proba, predict_proba, predict ** ** ** ** ** ** **

        for attr in (
            'decision_function', 'predict_log_proba', 'predict_proba', 'predict'
        ):

            if _X_state == 'good':  # da or df
                __ = getattr(dask_GSTCV, attr)(X_dask)
                assert isinstance(__, (np.ndarray, da.core.Array))
                if attr == 'predict':
                    assert __.dtype == np.uint8
                else:
                    assert __.dtype == np.float64
            elif _X_state == 'bad_features':
                if _X_format == 'da':
                    # this raises in the estimator, let is raise whatever
                    with pytest.raises(Exception):
                        getattr(dask_GSTCV, attr)(X_dask)
                elif _X_format == 'df':
                    # this raises in the estimator, let is raise whatever
                    with pytest.raises(Exception):
                        getattr(dask_GSTCV, attr)(X_dask)
            elif _X_state == 'bad_data':  # da or df
                # this raises in the estimator, let is raise whatever
                with pytest.raises(Exception):
                    getattr(dask_GSTCV, attr)(X_dask)
        # END decision_function, predict_log_proba, predict_proba, predict ** ** ** ** **





