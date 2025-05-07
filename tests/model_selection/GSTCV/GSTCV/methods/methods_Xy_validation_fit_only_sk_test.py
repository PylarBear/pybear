# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
from copy import deepcopy
import numpy as np
import pandas as pd


# this module tests fit for handling X & y in good and bad conditions
# (the only possible bad condition is non-num data, all other 'bad' things
# like num rows and num columns cant be bad first time passed)



class TestSKFit_XyValidation:


    # pizza this was hiding in validation_init_test
    # @pytest.mark.parametrize('junk_X',
    #     (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
    #      {'a': 1}, lambda x: x)
    # )
    # def test_rejects_junk_X(self, junk_X, y_np, base_gstcv):
    #
    #     # this is raised by GSTCV for no shape attr
    #     with pytest.raises(TypeError):
    #         base_gstcv.fit(junk_X, y_np)
    #
    #
    # @pytest.mark.parametrize('junk_y',
    #     (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
    #      {'a': 1}, lambda x: x)
    # )
    # def test_rejects_junk_y(self, X_np, junk_y, base_gstcv):
    #
    #     # this is raised by GSTCV for no shape attr
    #     with pytest.raises(TypeError):
    #         base_gstcv.fit(X_np, junk_y)




    @pytest.mark.parametrize('fit_format', ('array', 'df'))
    @pytest.mark.parametrize('_X_state', ('good', 'bad_data'))
    @pytest.mark.parametrize('_y_state', ('good', 'bad_data'))
    def test_fit(self, sk_est_log, fit_format, _X_state, _y_state,
        X_np, _rows, _cols, COLUMNS, non_binary_y, non_num_X,
        sk_GSTCV_est_log_one_scorer_prefit
    ):

        if _X_state == 'good' and _y_state == 'good':
            pytest.skip(reason=f'other tests already show fit on good works')


        # need to make a new instance of the prefit GSTCV, because the fitting
        # tests alter its state along the way, and it is a session fixture
        shallow_params = \
            deepcopy(sk_GSTCV_est_log_one_scorer_prefit.get_params(deep=False))
        deep_params = \
            deepcopy(sk_GSTCV_est_log_one_scorer_prefit.get_params(deep=True))

        sk_GSTCV = type(sk_GSTCV_est_log_one_scorer_prefit)(**shallow_params)
        sk_GSTCV.set_params(**deep_params)

        # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _X_state == 'good':
            X_sk = X_np
        elif _X_state == 'bad_data':
            X_sk = np.random.choice(list('abcd'), (_rows, _cols), replace=True)

        if fit_format == 'df':
            X_sk = pd.DataFrame(data=X_sk, columns=COLUMNS)
        # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _y_state == 'good':
            y_sk = np.random.randint(0, 2, (_rows, 1))
        elif _y_state == 'bad_data':
            y_sk = np.random.choice(list('abcd'), (_rows, 1), replace=True)

        if fit_format == 'df':
            y_sk = pd.DataFrame(data=y_sk, columns=['y1'])
        # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        if _X_state == 'good':
            if _y_state == 'good':
                pass  # skipped above
            elif _y_state == 'bad_data':
                # GSTCV should raise for not in [0,1]
                with pytest.raises(ValueError):
                    getattr(sk_GSTCV, 'fit')(X_sk, y_sk)

        elif _X_state == 'bad_data':
            # for all states of y
            # this is raised by estimator, let it raise whatever
            with pytest.raises(Exception):
                getattr(sk_GSTCV, 'fit')(X_sk, y_sk)








