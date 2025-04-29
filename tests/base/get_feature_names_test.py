# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.base._get_feature_names import get_feature_names

import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf
import polars as pl


import pytest




class TestGetFeatureNames:

    # as of 25_01_02 the landscape of headers in the python ecosystem:
    # -- numpy array, pandas series, dask array, scipy sparse never have
    #       a header and get_feature_names() should always return None.
    # -- pandas dataframe, dask series, dask dataframe
    #     -- when created with a valid header of strs get_features_names()
    #           will return that header
    #     -- when created without a header (constructed with the default
    #           header of numbers) get_features_names() will except
    #           for invalid header
    # -- polars dataframe
    #     -- when created with a valid header of strs get_features_names()
    #           will return that header
    #     -- when created without a header, constructed with a
    #           default header of STRINGS and get_features_names() will
    #           return that header



    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (37, 13)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]].astype(object)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_X_factory, _shape):
        return _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants=None,
            _zeros=0,
            _shape=_shape
        )


    @pytest.mark.parametrize('_format',
        ('np', 'pd_series', 'pd_df', 'csr_array', 'dask_array',
         'dask_series', 'dask_ddf', 'polars')
    )
    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_accuracy(
        self, _shape, _columns, _X_np, _format, _columns_is_passed
    ):

        if _format == 'np':
            _X_wip = _X_np
        elif _format == 'pd_series':
            _X_wip =pd.DataFrame(
                data = _X_np,
                columns = _columns if _columns_is_passed else None
            )
            _X_wip = _X_wip.iloc[:, 0].squeeze()
        elif _format == 'pd_df':
            _X_wip =pd.DataFrame(
                data = _X_np,
                columns = _columns if _columns_is_passed else None
            )
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X_np)
        elif _format == 'dask_array':
            _X_wip = da.from_array(_X_np, chunks=_shape)
        elif _format == 'dask_series':
            _X_wip = ddf.from_array(
                arr=_X_np,
                columns=_columns if _columns_is_passed else None,
                chunksize=_shape
            )
            _X_wip = _X_wip.iloc[:, 0].squeeze()
        elif _format == 'dask_ddf':
            _X_wip = ddf.from_array(
                arr=_X_np,
                columns=_columns if _columns_is_passed else None,
                chunksize=_shape
            )
        elif _format == 'polars':
            _X_wip = pl.DataFrame(
                data=_X_np,
                schema=_columns.tolist() if _columns_is_passed else None,
                orient='row'
            )
        else:
            raise Exception

        if _format in ['pd_df', 'dask_series', 'dask_ddf'] and \
                not _columns_is_passed:
            with pytest.warns():
                # this warns for non-str feature names
                # (the default header when 'columns=' is not passed)
                out = get_feature_names(_X_wip)
        else:
            out = get_feature_names(_X_wip)

        if not _columns_is_passed:
            if _format == 'polars':
                assert isinstance(out, np.ndarray)
                assert out.dtype == object
                assert np.array_equal(out, [f'column_{i}' for i in range(_shape[1])])
            else:
                assert out is None
        elif _columns_is_passed:
            if _format in ['pd_df', 'dask_series', 'dask_ddf', 'polars']:
                assert isinstance(out, np.ndarray)
                assert out.dtype == object
                if _format == 'dask_series':
                    assert np.array_equal(out, _columns[:1])
                else:
                    assert np.array_equal(out, _columns)
            else:
                assert out is None







