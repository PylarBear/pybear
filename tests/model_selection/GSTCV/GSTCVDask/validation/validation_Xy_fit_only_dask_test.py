# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
from copy import deepcopy
import dask.array as da
import dask.dataframe as ddf


# this module tests fit for handling X & y in good and bad conditions
# (the only possible bad condition is non-num data, all other 'bad' things
# like num rows and num columns cant be bad first time passed)



class TestDaskFit_XyValidation:


    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(
        self, junk_X, y_da, dask_GSTCV_est_log_one_scorer_prefit, _client
    ):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            dask_GSTCV_est_log_one_scorer_prefit.fit(junk_X, y_da)


    @pytest.mark.parametrize('junk_y',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_y(
        self, X_da, junk_y, dask_GSTCV_est_log_one_scorer_prefit, _client
    ):

        # this is raised by GSTCV for no shape attr
        with pytest.raises(TypeError):
            dask_GSTCV_est_log_one_scorer_prefit.fit(X_da, junk_y)


    @pytest.mark.parametrize('_container', ('da', 'df'))
    @pytest.mark.parametrize('_X_state', ('good', 'bad_data'))
    @pytest.mark.parametrize('_y_state', ('good', 'bad_data'))
    def test_data(
        self, _container, _X_state, _y_state, X_da, _rows, _cols,
        COLUMNS, dask_GSTCV_est_log_one_scorer_prefit, #_client
    ):

        if _X_state == 'good' and _y_state == 'good':
            pytest.skip(reason=f'other tests already show fit on good works')


        # need to make a new instance of the prefit GSTCV, because the fitting
        # tests alter its state along the way, and it is a session fixture
        shallow_params = \
            deepcopy(dask_GSTCV_est_log_one_scorer_prefit.get_params(deep=False))
        deep_params = \
            deepcopy(dask_GSTCV_est_log_one_scorer_prefit.get_params(deep=True))

        dask_GSTCV = type(dask_GSTCV_est_log_one_scorer_prefit)(**shallow_params)
        dask_GSTCV.set_params(**deep_params)

        # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _X_state == 'good':
            X_dask = X_da
        elif _X_state == 'bad_data':
            X_dask = da.random.choice(list('abcd'), (_rows, _cols), replace=True)

        if _container == 'df':
            X_dask = ddf.from_dask_array(X_dask, columns=COLUMNS)
        # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _y_state == 'good':
            y_dask = da.random.randint(0, 2, (_rows, 1))
        elif _y_state == 'bad_data':
            y_dask = da.random.choice(list('abcd'), (_rows, 1), replace=True)

        if _container == 'df':
            y_dask = ddf.from_dask_array(y_dask, columns=['y1'])
        # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if _container == 'df':
            with pytest.raises(TypeError):
                getattr(dask_GSTCV, 'fit')(X_dask, y_dask)
            pytest.skip(reason=f"25_04_29 GSTCVDask only accept dask array")

        if _X_state == 'good':
            if _y_state == 'good':
                pass  # skipped above
            elif _y_state == 'bad_data':
                # GSTCVDask should raise for not in [0,1]
                with pytest.raises(ValueError):
                    getattr(dask_GSTCV, 'fit')(X_dask, y_dask)

        elif _X_state == 'bad_data':
            # for all states of y
            # this is raised by estimator, let it raise whatever
            with pytest.raises(Exception):
                getattr(dask_GSTCV, 'fit')(X_dask, y_dask)





