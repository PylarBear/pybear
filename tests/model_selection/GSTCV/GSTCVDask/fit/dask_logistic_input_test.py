# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2


pytest.skip(reason=f'for edification purposes only', allow_module_level=True)
# no need to test the volatile and unpredictable state of dask error messages

class TestDaskLogistic:


    # 24_06_28_11_39_00 there are a lot of ifs ands and buts about what
    # dask_Logistic.fit() can take for X and y, but the simplest distillation
    # is:
    # 1) cannot ever take pd nor dask DFs for X, no add_intercept attr
    # 2) cannot ever take pd nor dask Series for X, no add_intercept attr
    # 3) cannot ever take pd nor dask DFs for y, no ravel attr
    # 4) cannot mix dask and non-dask Xs & ys due to chunk mismatches

    # so the simplest way to robustly pass X and y to logistic fit is
    # both are either numpy array or dask array

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def X_dask_series(X_ddf):
        assert isinstance(X_ddf, (ddf.core.DataFrame, ddf2.DataFrame))
        return X_ddf.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def X_pd_series(X_pd):
        assert isinstance(X_pd, pd.core.frame.DataFrame)
        return X_pd.iloc[:, 0]


    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def y_dask_series(y_ddf):
        assert isinstance(y_ddf, (ddf.core.DataFrame, ddf2.DataFrame))
        return y_ddf.iloc[:, 0]


    @staticmethod
    @pytest.fixture
    def y_pd_series(y_pd):
        assert isinstance(y_pd, pd.core.frame.DataFrame)
        return y_pd.iloc[:, 0]

    # END y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # tests ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_y_format',
        ('np_array', 'pd_df', 'pd_series', 'dask_array', 'dask_df', 'dask_series')
    )
    def test_passing_X_y_in_different_formats(self, dask_est_log, X_da, X_ddf,
        X_dask_series, X_np, X_pd, X_pd_series, _y_format, y_da, y_ddf,
        y_dask_series, y_np, y_pd, y_pd_series
    ):

        if _y_format == 'np_array':
            _y = y_np
        elif _y_format == 'pd_df':
            _y = y_pd
        elif _y_format == 'pd_series':
            _y = y_pd_series
        elif _y_format == 'dask_array':
            _y = y_da
        elif _y_format == 'dask_df':
            _y = y_ddf
        elif _y_format == 'dask_series':
            _y = y_dask_series


    # make of copy of dask_est_log, these fits will stick since its session scope
        dask_est = type(dask_est_log)()


    # test_np_array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * 

        # y cannot be pd.DF, dask.DF for no ravel() attribute
        # y cannot be da.Array because chunks dont match

        if isinstance(_y,
            (pd.core.frame.DataFrame, ddf.core.DataFrame, ddf2.DataFrame)):
            with pytest.raises(AttributeError):
                dask_est.fit(X_np, _y)
        elif isinstance(_y, da.core.Array):
            with pytest.raises(ValueError):
                dask_est.fit(X_np, _y)
        else:
            dask_est.fit(X_np, _y)

    # END test_np_array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test_pd_df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # NotImplementedError: Could not find signature for add_intercept: <DataFrame>

        with pytest.raises(NotImplementedError):
            dask_est.fit(X_pd, _y)

    # END test_pd_df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test_pd_series ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # NotImplementedError: Could not find signature for add_intercept: <Series>

        with pytest.raises(NotImplementedError):
            dask_est.fit(X_pd_series, _y)

    # END test_pd_series ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # test_dask_array ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if isinstance(_y,
            (np.ndarray, pd.core.series.Series, ddf.core.Series, ddf2.Series)):
            # chunks dont match
            with pytest.raises(ValueError):
                dask_est.fit(X_da, _y)
        elif isinstance(_y,
            (pd.core.frame.DataFrame, ddf.core.DataFrame, ddf2.DataFrame)):
            # no ravel() attribute
            with pytest.raises(AttributeError):
                dask_est.fit(X_da, _y)
        else:
            dask_est.fit(X_da, _y)

    # END test_dask_array ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test_dask_df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # TypeError: This estimator does not support dask dataframes.
        # This might be resolved with one of
        #   1. ddf.to_dask_array(lengths=True)
        #   2. ddf.to_dask_array()  # may cause other issues because of
        #       unknown chunk sizes

        with pytest.raises(TypeError):
            dask_est_log.fit(X_ddf, _y)

    # END test_dask_df ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test_dask_series ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # E       AttributeError: 'Series' object has no attribute 'columns'

        with pytest.raises(AttributeError):
            dask_est.fit(X_dask_series, _y)

    # END test_dask_series ** * ** * ** * ** * ** * ** * ** * ** * ** *












